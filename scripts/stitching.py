import os
import numpy as np
import torch
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from pycolmap import Reconstruction
from hloc.utils import viz_3d
from hloc.utils.io import get_matches
import torchvision
import pycolmap
import cv2
import h5py


def create_sfm_model(image_dir, output_path, references, feature_type="disk", matcher="disk+lightglue", visualize=False, feature_conf=None):
  """
  Generate features for a set of images and perform matching.

  Args:
    image_dir (Path): Path to the directory containing the input images.
    references. List of paths to the input images. Can be a subset of the images in the directory.
    output_path (Path): Path to the output directory.
    feature_type (str, optional): Type of feature extraction method. Defaults to "disk".
    matcher (str, optional): Type of feature matching method. Defaults to "disk+lightglue".
    visualize (bool, optional): Whether to visualize the SfM model. Defaults to False.
    feature_conf (dict, optional): Configuration for feature extraction. Defaults to None.

  Returns:
    None
  """
  sfm_pairs = output_path / 'pairs-sfm.txt'
  loc_pairs = output_path / 'pairs-from-sfm.txt'
  sfm_dir = output_path / 'sfm'
  features = output_path / 'features.h5'
  matches = output_path / 'matches.h5'

  # Set feature type
  if feature_conf is None:
    feature_conf = extract_features.confs[feature_type]
  matcher_conf = match_features.confs[matcher]

  # Match and write files to disk
  if not os.path.exists(features):
    extract_features.main(conf=feature_conf, image_dir=image_dir, image_list=references, feature_path=features)
  if not os.path.exists(sfm_pairs):
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
  if not os.path.exists(matches):
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
  #if not os.path.exists(sfm_dir):
  #  model = reconstruction.main(sfm_dir = sfm_dir, image_dir=image_dir, pairs=sfm_pairs, features=features, matches=matches)
  #else:
  #  model = Reconstruction(sfm_dir)
  
  #if model is None:
  #  raise ValueError("SfM model failed to reconstruct")
  
  #if visualize:
  #  visualization.visualize_sfm_2d(model, image_dir, color_by="visibility", n=1)

  #return model

def get_matching_points(h5_file, image1_name, image2_name, min_score=None):
  """
  Get matching points between two images from an h5 file.

  Args:
    h5_file (str): Path to the h5 file containing the matches.
    image1_name (str): Name of the first image.
    image2_name (str): Name of the second image.
    min_score (float, optional): Minimum score for a match to be considered. Defaults to None

  Returns:
    tuple: Two numpy arrays containing the matching points from the two images.
  """
  matches, scores = get_matches(h5_file, image1_name, image2_name)
  if min_score is not None:
    matches = matches[scores > min_score]
  match_index = pd.DataFrame(matches, columns=["image1", "image2"])
  
  # Open features h5 and lookup matching features
  with h5py.File(os.path.dirname(h5_file) + "/features.h5", 'r') as features_h5_f:
    keypoints_image1 = pd.DataFrame(features_h5_f[image1_name]["keypoints"][:], columns=["x", "y"])
    keypoints_image2 = pd.DataFrame(features_h5_f[image2_name]["keypoints"][:], columns=["x", "y"])
    points1 = keypoints_image1.iloc[match_index["image1"].values].values
    points2 = keypoints_image2.iloc[match_index["image2"].values].values

  return points1, points2

def compute_homography_matrix(model, h5_file, image1_name, image2_name):
  """Compute the homography matrix between two images.
  Args:
      h5_file (str): Path to the h5 file containing the matches.
      image1 (str): Name of the first image.
      image2 (str): Name of the second image.
  Returns:
      dict: Dictionary of estimation outputs or None if failure.
  """
  points1, points2 = get_matching_points(h5_file, image1_name, image2_name)

  # Raise error if points are empty
  if len(points1) == 0 or len(points2) == 0:
    raise ValueError("No matching points found between images {} and {}".format(image1_name, image2_name))
  
  m = pycolmap.homography_matrix_estimation(points1, points2)

  if m is None:
    raise ValueError("Homography matrix estimation failed for images {} and {}".format(image1_name, image2_name))
  
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

def remove_predictions(src_predictions, dst_predictions, aligned_predictions, threshold=0.1, strategy='highest-score'):
  """
  Remove overlapping predictions using non-max suppression.

  Args:
    predictions (DataFrame): A pandas DataFrame containing aligned prediction data. Must have a box_id column that matches original and aligned predictions.
    threshold (float, optional): The threshold value for non-max suppression. Default is 0.1.
    strategy (str, optional): The strategy to use to remove duplicate detection, 'Highest-score' selects the better scoring box, 'left-hand' selects the box from the earlier image, 'right-hand' selects the box from the later image. Default is 'highest_score'.

  Returns:
    DataFrame: A filtered DataFrame containing non-overlapping predictions.
  """
  dst_and_aligned_predictions = pd.concat([aligned_predictions, dst_predictions])
  if strategy == "highest-score":
    # Perform non-max suppression on aligned predictions
    # Convert bounding box coordinates to torch tensors
    boxes = torch.tensor(dst_and_aligned_predictions[['xmin', 'ymin', 'xmax', 'ymax']].values)

    # Convert scores to torch tensor
    scores = torch.tensor(dst_and_aligned_predictions['score'].values)

    # Perform non-max suppression
    keep = torchvision.ops.nms(boxes, scores, threshold)
    indices_to_keep = dst_and_aligned_predictions.iloc[keep]
    
    #Split into source and destination
    src_filtered = src_predictions[src_predictions.box_id.isin(indices_to_keep.box_id)]
    dst_filtered = dst_predictions[dst_predictions.box_id.isin(indices_to_keep.box_id)]

  else:
    aligned_predictions["box"] = aligned_predictions.apply(lambda row: box(row['xmin'], row['ymin'], row['xmax'], row['ymax']), axis=1)
    dst_predictions["box"] = dst_predictions.apply(lambda row: box(row['xmin'], row['ymin'], row['xmax'], row['ymax']), axis=1)
    aligned_gdf = gpd.GeoDataFrame(aligned_predictions, geometry="box")
    dst_gdf = gpd.GeoDataFrame(dst_predictions, geometry='box')

    # Join the two dataframes
    joined = gpd.sjoin(aligned_gdf, dst_gdf, how='inner', predicate='intersects')

    if strategy == "left-hand":
      # Where there is overlap, remove the right hand image
      src_indices_to_keep = src_predictions.box_id
      dst_indices_to_keep = dst_predictions[~dst_predictions.box_id.isin(joined.box_id_right)].box_id
    else:
      # Where there is overlap, remove the left hand image
      src_indices_to_keep = src_predictions[~src_predictions.box_id.isin(joined.box_id_left)].box_id
      dst_indices_to_keep = dst_predictions.box_id

    src_filtered = src_predictions[src_predictions.box_id.isin(src_indices_to_keep)]
    dst_filtered = dst_predictions[dst_predictions.box_id.isin(dst_indices_to_keep)]

  return src_filtered, dst_filtered


def align_and_delete(model, matching_h5_file, predictions, threshold=0.3, image_dir=None, strategy='highest_score'):
  """
  Given a set of images and predictions, align the images using the sfm_model and delete overlapping images.

  Args:
    model (SfMModel): The SfM model containing the images.
    predictions (DataFrame): The predictions dataframe containing the bounding box predictions.
    threshold (float, optional): The threshold value for non-max suppression. Defaults to 0.5.
    image_dir (Path, optional): Path to the directory containing the images. Defaults to None.

  Returns:
    DataFrame: The filtered predictions dataframe after aligning and deleting overlapping images.
  """
  # Load the SfM model  
  image_names = predictions.image_path.unique()
  image_names.sort()

  # Make sure the indices are reset and unique
  predictions["box_id"] = predictions.reset_index().index

  # Create a dictionary of prediction to filter
  filtered_predictions = {}
  for x in image_names:
    filtered_predictions[x] = predictions[predictions.image_path == x]

  for index, src_image_name in enumerate(image_names):
    for dst_image_name in image_names[index+1:]:
      # Compute homography   
      try:
        homography = compute_homography_matrix(model=model, h5_file=matching_h5_file, image1_name=src_image_name, image2_name=dst_image_name)
      except Exception:
        continue

      # Get the predictions for the source and destination images
      src_image_predictions = filtered_predictions[src_image_name]
      dst_image_predictions = filtered_predictions[dst_image_name]
      
      # If the predictions are empty, skip
      if src_image_predictions.empty or dst_image_predictions.empty:
        continue

      # Align and remove
      aligned_predictions = align_predictions(predictions=src_image_predictions, homography_matrix=homography["H"])   
      src_filtered_predictions, dst_filtered_predictions = remove_predictions(
        src_predictions=src_image_predictions,
        dst_predictions=dst_image_predictions,
        aligned_predictions=aligned_predictions,
        threshold=threshold,
        strategy=strategy)
      
      filtered_predictions[src_image_name] = src_filtered_predictions
      filtered_predictions[dst_image_name] = dst_filtered_predictions
  
  # Concatenate the filtered predictions
  filtered_predictions = pd.concat(filtered_predictions.values())

  return filtered_predictions

def collect_keypoints(predictions, matching_h5_file):
  image_names = predictions.image_path.unique()
  image_names.sort()

  keypoints_data = []

  for index, src_image_name in enumerate(image_names):
    for dst_image_name in image_names[index+1:]:
      keypoints = generate_keypoints(matching_h5_file, src_image_name, dst_image_name)
      if keypoints is None:
        continue
      keypoints_data.append(keypoints)
  
  keypoints_data = pd.concat(keypoints_data)

  return keypoints_data

def generate_keypoints(matching_h5_file, image1_name, image2_name, min_match_score=0.95):
  """
  Generate a DataFrame of keypoints with random colors for each unique match.

  Args:
    matching_h5_file (str): Path to the h5 file containing the matches.
    image1_name (str): Name of the first image.
    image2_name (str): Name of the second image.
    min_match_score (float, optional): Minimum score for a match to be considered. Defaults to 0.95.

  Returns:
    DataFrame: A DataFrame with columns x, y, image
  """
  points1, points2 = get_matching_points(matching_h5_file, image1_name, image2_name, min_score=min_match_score)
  keypoints_list = []
  for index, ((x1, y1), (x2, y2)) in enumerate(zip(points1, points2)):

    keypoints_list.append({'x': x1, 'y': y1, 'image': image1_name, 'match_image': image2_name})
    keypoints_list.append({'x': x2, 'y': y2, 'image': image2_name, 'match_image': image1_name})

  # Create a DataFrame from the keypoints list
  keypoints_df = pd.DataFrame(keypoints_list)

  if keypoints_df.empty:
    return None
  
  return keypoints_df

