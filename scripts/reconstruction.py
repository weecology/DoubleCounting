# 3D reconstruction from 2D images using colmap and pycolmap
import numpy as np
import pycolmap

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
