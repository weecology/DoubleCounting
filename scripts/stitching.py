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

def transform_2d_to_3d(x, y, intrinsic, extrinsic, depth_map=None, scalar_depth_range=None):
  """Convert a 2d point to a 3d point using the camera intrinsic and extrinsic parameters.
  Inspired by https://github.com/Totoro97/f2-nerf/blob/98f0daacb80e76724eb91519742c30fb35d0f72d/scripts/colmap2poses.py#L58
  https://github.com/colmap/colmap/issues/1476
  https://github.com/colmap/colmap/issues/1424
  https://github.com/colmap/colmap/issues/797

  Args:
      x (numeric): x location of the point
      y (numeric): y location of the point
      intrinsic (np.array): colmap representation of intrinsic matrix in the format [f, cx, cy, k]
      extrinsic (pycolmap extrinsic matrix ): cam_to_world representation of the extrinsic matrix for a given image to the reconstruction
      depth_map (np.array, optional): The depth map of the image. Defaults to None.
      scalar_depth_range (list, optional): The range of depths to consider. Defaults to [0,10].
  Returns:
      world_direction_vector (np.array): The 3D point in the world coordinate system.
  """
  # Add a dummy dimension to the 2D point for Z
  p = np.array([x, y, 1]).T

  if depth_map is not None:
    # Get the depth value
    depth = depth_map[int(y), int(x)]
  else:
    depth = 1
  
  # Create intrinsic matrix from the colmap format
  intrinsic_matrix = create_intrinsic_matrix(f=intrinsic[0], cx=intrinsic[1], cy=intrinsic[2], k=0)
  
  # colmap format is world to camera, so we need to invert the matrix
  rotation_matrix = extrinsic.rotation.matrix().T
  translation = -rotation_matrix.dot(extrinsic.translation)
  inverted_intrinsic = np.linalg.inv(intrinsic_matrix)

  # Transform pixel in Camera coordinate frame to World coordinate frame
  direction_vector = inverted_intrinsic.dot(p)
  
  # Normalize the direction vector
  direction_vector = direction_vector / np.linalg.norm(direction_vector)
  world_direction_vector =  (rotation_matrix.dot(direction_vector))

  # Get Camera origin
  cam = np.array([0,0,0]).T
  cam_world = translation + (rotation_matrix @ cam)

  points3D = []
  if scalar_depth_range:
    for depth in range(scalar_depth_range[0], scalar_depth_range[1]):
      world_position = translation +  world_direction_vector * depth
      points3D.append(world_position)
  else:
    world_position = translation + world_direction_vector * depth
    world_position = world_position / np.linalg.norm(world_position)
    points3D.append(world_position)
      
  return points3D

def create_intrinsic_matrix(f, cx, cy, k=0):
  """
  Creates an intrinsic camera matrix from focal length, principal point, and distortion coefficient. With some inspiration from Google Gemini. 

  Args:
      f (float): Focal length of the camera.
      cx (float): X-coordinate of the principal point.
      cy (float): Y-coordinate of the principal point.
      k (float, optional): Distortion coefficient (default: 0).

  Returns:
      numpy.ndarray: The 3x3 intrinsic camera matrix.
  """
  # Create the base matrix with zeros
  intrinsic_matrix = np.zeros((3, 3))

  # Assign focal length values
  intrinsic_matrix[0, 0] = f
  intrinsic_matrix[1, 1] = f

  # Set principal point coordinates
  intrinsic_matrix[0, 2] = cx
  intrinsic_matrix[1, 2] = cy

  # Add skew factor if provided (typically assumed to be 0)
  intrinsic_matrix[1, 0] = k  # Adjust based on your distortion model # This should be 0.

  # Set the last element to 1
  intrinsic_matrix[2, 2] = 1

  return intrinsic_matrix

def align_prediction(predictions, intrinsic, extrinsic, depth_map=None):
  """Align the predictions to the SfM model.
  For camera intrinsic parameters defined in calibration matrix K(3,3), Transformation matrix, M = [R | t] (3,4)

  Args:
    predictions (DataFrame): The predictions dataframe containing the bounding box predictions.
    intrinsic (array): The intrinsic camera matrix.
    extrinsic (array): The extrinsic camera matrix.
  
  Returns:
    DataFrame: The aligned predictions.
  """
  transformed_predictions = predictions.copy(deep=True)
  for index, row in transformed_predictions.iterrows():
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']
    transformed_xmin, transformed_ymin, transformed_zmin = transform_2d_to_3d(xmin, ymin, intrinsic, extrinsic, depth_map)
    transformed_xmax, transformed_ymax, transformed_zmax = transform_2d_to_3d(xmax, ymax, intrinsic, extrinsic, depth_map)
    x_centroid = (xmin + xmax) / 2
    y_centroid = (ymin + ymax) / 2
    transformed_x_centroid, transformed_y_centroid, transformed_z_centroid = transform_2d_to_3d(x_centroid, y_centroid, intrinsic, extrinsic)

    # Update the transformed predictions
    transformed_predictions.at[index, 'xmin'] = transformed_xmin
    transformed_predictions.at[index, 'ymin'] = transformed_ymin
    transformed_predictions.at[index, 'xmax'] = transformed_xmax
    transformed_predictions.at[index, 'ymax'] = transformed_ymax

    transformed_predictions.at[index, 'x_centroid'] = transformed_x_centroid
    transformed_predictions.at[index, 'y_centroid'] = transformed_y_centroid
    transformed_predictions.at[index, 'z_centroid'] = transformed_z_centroid
  
  return transformed_predictions

def align_and_delete(model, predictions, threshold=0.5, image_dir=None, visualize=False):
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
  references_registered = [model.images[i].name for i in model.reg_image_ids()]
  image_names = predictions.image_path.unique()

  # Intrinsic camera matrix
  intrinsic = model.cameras[1].params
  camera = model.cameras[1]

  aligned_predictions_across_images = []
  for image_name in image_names:
    image_predictions = predictions[predictions.image_path == image_name]
    references_registered = [model.images[i].name for i in model.reg_image_ids()]
    try:
      image_index = references_registered.index(image_name)
      # Careful with the indexing here, as the image_index is based on the registered images, not sorted.
      model_index = model.reg_image_ids()[image_index]
    except ValueError:
      continue
    image = model.images[model_index]
    extrinsic = image.cam_from_world

    # Align the predictions
    aligned_predictions = align_prediction(image_predictions, intrinsic, extrinsic, depth_map=None)
    aligned_predictions_across_images.append(aligned_predictions) 
  
  aligned_predictions_across_images = pd.concat(aligned_predictions_across_images)

  # View the aligned predictions using geopandas
  gdf = gpd.GeoDataFrame(aligned_predictions_across_images,
                        geometry=gpd.points_from_xy(aligned_predictions_across_images.x_centroid,
                                                     aligned_predictions_across_images.y_centroid))
  # color by image name
  if visualize:
    gdf.plot(column='image_path', figsize=(10, 10))
    pyplot.show()

    # View aligned bounding boxes
    gdf['box'] = gdf.apply(lambda row: box(row['xmin'], row['ymin'], row['xmax'], row['ymax']), axis=1)
    gdf = gpd.GeoDataFrame(gdf, geometry='box')
    gdf.plot(column='image_path', figsize=(10, 10))
    pyplot.show()

    #plot images and keypoints
    # plot_images([read_image(image_dir / x) for x in image_names], dpi=200)
    # keypoints = []
    # for image_name in image_names:
    #   image_predictions = predictions[predictions.image_path == image_name]
    #   keypoints.append(image_predictions[['x_centroid', 'y_centroid']].values)
    # plot_keypoints(keypoints)

    # Plot the images and keypoints
    visualization.visualize_sfm_2d(model, image_dir, color_by='visibility', n=2)   
    
    fig = viz_3d.init_figure() 
    viz_3d.plot_reconstruction(fig, model)
    fig.show()

  # Perform non-max suppression on aligned predictions
  # Convert bounding box coordinates to torch tensors
  boxes = torch.tensor(aligned_predictions_across_images[['xmin', 'ymin', 'xmax', 'ymax']].values)

  # Convert scores to torch tensor
  scores = torch.tensor(aligned_predictions_across_images['score'].values)

  # Perform non-max suppression
  keep = torchvision.ops.nms(boxes, scores, threshold)

  # Filter the original dataframe based on the keep indices
  filtered_df = predictions.iloc[keep]

  return filtered_df