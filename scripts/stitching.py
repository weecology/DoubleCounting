import os
import numpy as np
import torch
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, triangulation
from pycolmap import Reconstruction
from hloc.utils import viz_3d
from hloc.utils.io import get_matches
import torchvision
import pycolmap
import cv2

import matplotlib.pyplot as pyplot

def create_sfm_model(image_dir, output_path, references, feature_type="disk", matcher="disk+lightglue", visualize=False):
  """
  Generate features for a set of images and perform matching.

  Args:
    image_dir (Path): Path to the directory containing the input images.
    references. List of paths to the input images. Can be a subset of the images in the directory.
    output_path (Path): Path to the output directory.
    feature_type (str, optional): Type of feature extraction method. Defaults to "disk".
    matcher (str, optional): Type of feature matching method. Defaults to "disk+lightglue".
    visualize (bool, optional): Whether to visualize the SfM model. Defaults to False.

  Returns:
    None
  """
  sfm_pairs = output_path / 'pairs-sfm.txt'
  loc_pairs = output_path / 'pairs-from-sfm.txt'
  sfm_dir = output_path / 'sfm'
  features = output_path / 'features.h5'
  matches = output_path / 'matches.h5'
  results = output_path / 'results.txt'
  reference_sfm = output_path / "sfm_superpoint+superglue"  # the SfM model we will build
  database_path = output_path / "database.db"
  mvs_path = output_path / "mvs"

  # Set feature type
  feature_conf = extract_features.confs[feature_type]
  matcher_conf = match_features.confs[matcher]

  # Match and write files to disk
  if not os.path.exists(features):
    extract_features.main(conf=feature_conf, image_dir=image_dir, image_list=references, feature_path=features)
  if not os.path.exists(sfm_pairs):
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
  if not os.path.exists(matches):
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
  if not os.path.exists(sfm_dir):
    model = reconstruction.main(sfm_dir = sfm_dir, image_dir=image_dir, pairs=sfm_pairs, features=features, matches=matches)
  else:
    model = Reconstruction(sfm_dir)
  
  if visualize:
    visualization.visualize_sfm_2d(model, image_dir, color_by="visibility", n=1)

  return model

def compute_homography_matrix(model, h5_file, image1_name, image2_name):
  """Compute the homography matrix between two images.
  Args:
      h5_file (str): Path to the h5 file containing the matches.
      image1 (str): Name of the first image.
      image2 (str): Name of the second image.
  Returns:
      dict: Dictionary of estimation outputs or None if failure.
  """
  matches, scores = get_matches(h5_file, image1_name, image2_name)
  image1 = model.find_image_with_name(image1_name)
  image2 = model.find_image_with_name(image2_name)

  # Look up the matching points for each index, which is the first and which is the second index?
  points1 = [image1.points2D[i].xy for i in matches[:,0]]
  points2 = [image2.points2D[i].xy for i in matches[:,1]]

  m = pycolmap.homography_matrix_estimation(points1, points2)

  return m

def warp_points(homography_matrix, points):
  """Warp a set of points using a homography matrix.
  Args:
      homography_matrix (np.array): The homography matrix.
      points (np.array): The points to warp.
  Returns:
      np.array: The warped points.
  """
  # As float, add dummy z 
  reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
  warped_points = cv2.perspectiveTransform(reshaped_points, homography_matrix)

  return warped_points

def warp_box(xmin, xmax, ymin, ymax, homography):
  """Warp a bounding box using a homography matrix.
  Args:
      xmin (float): The minimum x-coordinate of the bounding box.
      xmax (float): The maximum x-coordinate of the bounding box.
      ymin (float): The minimum y-coordinate of the bounding box.
      ymax (float): The maximum y-coordinate of the bounding box.
      homography (np.array): The homography matrix.
  Returns:
      tuple: The warped bounding box coordinates.
  """
  points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
  warped_points = warp_points(homography, points)
  # Opencv pads the dimension in a unnecessary way
  warped_points = warped_points.squeeze(1).astype(int)
  warped_xmin = warped_points[:,0].min()
  warped_xmax = warped_points[:,0].max()
  warped_ymin = warped_points[:,1].min()
  warped_ymax = warped_points[:,1].max()

  return warped_xmin, warped_xmax, warped_ymin, warped_ymax

def align_predictions(predictions, homography_matrix):
  """Align the predictions to the SfM model.
  
  This function takes in a DataFrame of predictions containing bounding box coordinates and aligns them to the SfM (Structure from Motion) model using a homography matrix.
  
  Args:
    predictions (DataFrame): The predictions DataFrame containing the bounding box predictions.
    homography_matrix (array): The homography matrix used for alignment.
  
  Returns:
    DataFrame: The aligned predictions DataFrame.
  """
  transformed_predictions = predictions.copy(deep=True)
  for index, row in transformed_predictions.iterrows():
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']
    warp_boxed = warp_box(xmin, xmax, ymin, ymax, homography_matrix)
    transformed_predictions.loc[index, 'xmin'] = warp_boxed[0]
    transformed_predictions.loc[index, 'xmax'] = warp_boxed[1]
    transformed_predictions.loc[index, 'ymin'] = warp_boxed[2]
    transformed_predictions.loc[index, 'ymax'] = warp_boxed[3]

  return transformed_predictions

def remove_predictions(predictions, threshold=0.1, strategy='highest-score'):
  """
  Remove overlapping predictions using non-max suppression.

  Args:
    predictions (DataFrame): A pandas DataFrame containing prediction data.
    threshold (float, optional): The threshold value for non-max suppression. Default is 0.1.
    strategy (str, optional): The strategy to use to remove duplicate detection, 'Highest-score' selects the better scoring box, 'left-hand' selects the box from the earlier image, 'right-hand' selects the box from the later image. Default is 'highest_score'.

  Returns:
    DataFrame: A filtered DataFrame containing non-overlapping predictions.
  """
  if strategy == "highest-score":
    # Perform non-max suppression on aligned predictions
    # Convert bounding box coordinates to torch tensors
    boxes = torch.tensor(predictions[['xmin', 'ymin', 'xmax', 'ymax']].values)

    # Convert scores to torch tensor
    scores = torch.tensor(predictions['score'].values)

    # Perform non-max suppression
    keep = torchvision.ops.nms(boxes, scores, threshold)

    # Filter the original dataframe based on the keep indices
    filtered_df = predictions.iloc[keep]
  else:
    left_hand_image = predictions[predictions.image_path == predictions.image_path.unique()[0]]
    right_hand_image = predictions[predictions.image_path == predictions.image_path.unique()[1]]
    left_hand_image["box"] = left_hand_image.apply(lambda row: box(row['xmin'], row['ymin'], row['xmax'], row['ymax']), axis=1)
    right_hand_image["box"] = right_hand_image.apply(lambda row: box(row['xmin'], row['ymin'], row['xmax'], row['ymax']), axis=1)
    left_hand_image = gpd.GeoDataFrame(left_hand_image, geometry="box")
    right_hand_image = gpd.GeoDataFrame(right_hand_image, geometry='box')

    # Join the two dataframes
    joined = gpd.sjoin(left_hand_image, right_hand_image, how='inner', op='intersects')

    if strategy == "left-hand":
      # Where there is overlap, remove the right hand image
      filtered_df = predictions[~predictions.index.isin(joined.index_right)]
    else:
      # Where there is overlap, remove the left hand image
      filtered_df = predictions[~predictions.index.isin(joined.index_left)]
  
  return filtered_df


def align_and_delete(model, matching_h5_file, predictions, threshold=0.5, image_dir=None, visualize=True, strategy='highest_score'):
  """
  Given a set of images and predictions, align the images using the sfm_model and delete overlapping images.

  Args:
    model (SfMModel): The SfM model containing the images.
    predictions (DataFrame): The predictions dataframe containing the bounding box predictions.
    threshold (float, optional): The threshold value for non-max suppression. Defaults to 0.5.
    image_dir (Path, optional): Path to the directory containing the images. Defaults to None.
    visualize (bool, optional): Whether to visualize the aligned predictions. Defaults to False.

  Returns:
    DataFrame: The filtered predictions dataframe after aligning and deleting overlapping images.
  """
  # Load the SfM model  
  image_names = predictions.image_path.unique()
  image_names.sort()

  # For each image, align with the next image
  aligned_predictions_across_images = []
  for index, image_name in enumerate(image_names[:-1]):
    src = image_name
    dst = image_names[index + 1]
    
    # Compute homography   
    src_image_name = predictions.image_path.unique()[0]
    dst_image_name = predictions.image_path.unique()[1]
    homography = compute_homography_matrix(model=model, h5_file=matching_h5_file, image1_name=src_image_name, image2_name=dst_image_name)
    src_image_predictions = predictions[predictions.image_path == src_image_name]
    
    aligned_predictions = align_predictions(predictions=src_image_predictions, homography_matrix=homography["H"])

    # Combine predictions in source and dst images for removal
    predictions_to_remove = pd.concat([aligned_predictions, predictions[predictions.image_path == dst]])
    remaining_predictions = remove_predictions(predictions_to_remove, threshold=threshold, strategy=strategy)

    # Remove overlapping predictions
    aligned_predictions_across_images.append(remaining_predictions) 
  
  aligned_predictions_across_images = pd.concat(aligned_predictions_across_images)

  # color by image name
  if visualize:  
    # View aligned bounding boxes
    aligned_predictions_across_images['box'] = aligned_predictions_across_images.apply(lambda row: box(row['xmin'], row['ymin'], row['xmax'], row['ymax']), axis=1)
    gdf = gpd.GeoDataFrame(aligned_predictions_across_images, geometry='box')
    gdf.plot(column='image_path', figsize=(10, 10))
    pyplot.show()

  return aligned_predictions_across_images