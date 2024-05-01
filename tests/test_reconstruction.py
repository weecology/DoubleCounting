from scripts.reconstruction import transform_2d_to_3d
from scripts.predict import predict
from scripts.stitching import create_sfm_model
import os
import glob
from pathlib import Path
import pandas as pd
import pytest
import numpy as np
import matplotlib.pyplot as pyplot
from hloc.visualization import plot_images, read_image, plot_keypoints

@pytest.fixture()
def test_data_dir():
    return os.path.join(os.path.dirname(__file__),"data")

@pytest.fixture
def model_simple(test_data_dir):
    output_path = Path(os.path.join(test_data_dir, "output_simple"))
    output_path.mkdir(parents=True, exist_ok=True)
    image_dir = Path(os.path.join(os.path.dirname(__file__),"data/simple"))
    references = [str(p.relative_to(image_dir)) for p in (image_dir).iterdir()]
    return create_sfm_model(image_dir=image_dir, references=references, output_path=output_path)

def test_transform_2d_to_3d_simple(test_data_dir, model_simple):
    # Create a test case for a simple images for multiple cameras. Point to the top of the spire in the church image and get the 3D coordinates in two images, they should overlap
    image_dir = Path(os.path.join(os.path.dirname(__file__),"data/simple"))

    # Coordinates of the top spire for simple1.JPG
    top_spire_image1_x = 510
    top_spire_image1_y = 51

    # Coordinates of the top spire for simple2.JPG
    top_spire_image2_x = 504 
    top_spire_image2_y = 28

    # Coordinates of the left spire for simple1.JPG
    left_spire_image1_x = 340
    left_spire_image1_y = 297

    # Coordinates of the left spire for simple2.JPG
    left_spire_image2_x = 383 
    left_spire_image2_y = 205

    image_dir = Path(os.path.join(os.path.dirname(__file__),"data/simple"))
    output_path = Path(os.path.join(test_data_dir, "output_simple"))
    output_path.mkdir(parents=True, exist_ok=True)
    references = [str(p.relative_to(image_dir)) for p in (image_dir).iterdir()]

    # List and view images
    images = [read_image(image_dir / x) for x in references]
    plot_images(images, dpi=200)
    # Plot keypoints of top and left spire
    keypoints = [np.array([[top_spire_image1_x, top_spire_image1_y], [left_spire_image1_x, left_spire_image1_y]]), np.array([[top_spire_image2_x, top_spire_image2_y], [left_spire_image2_x, left_spire_image2_y]])]
    plot_keypoints(keypoints)

    # Call the function with the proper index
    image_1 = model_simple.find_image_with_name("simple1.jpg")
    intrinsic_1 = model_simple.cameras[2].params
    extrinsic_1 = image_1.cam_from_world

    image_2 = model_simple.find_image_with_name("simple2.jpg")
    intrinsic_2 = model_simple.cameras[1].params
    extrinsic_2 =image_2.cam_from_world

    # Top spire
    result_1 = transform_2d_to_3d(top_spire_image1_x, top_spire_image1_y, intrinsic_1, extrinsic_1, scalar_depth_range=[0,100])
    result_2 = transform_2d_to_3d(top_spire_image2_x, top_spire_image2_y, intrinsic_2, extrinsic_2, scalar_depth_range=[0,100])

    # Left spire
    result_3 = transform_2d_to_3d(left_spire_image1_x, left_spire_image2_y, intrinsic_1, extrinsic_1, scalar_depth_range=[0,100])
    result_4 = transform_2d_to_3d(left_spire_image2_x, left_spire_image2_y, intrinsic_2, extrinsic_2, scalar_depth_range=[0,100])

    # View the output in matplotlib, plot the x_centroid, y_centroid, z_centroid coordinates
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Show as lines along the camera view
    for point in result_1:
        ax.scatter(point[0], point[1], point[2], c="red", label="Top Spire Image 1", s=5)
    for point in result_2:
        ax.scatter(point[0], point[1], point[2], c="blue", label="Top Spire Image 2", s=5)
    for point in result_3:
        ax.scatter(point[0], point[1], point[2], c="red", label="Left Spire Image 1", s=5)
    for point in result_4:
        ax.scatter(point[0], point[1], point[2], c="blue", label="Left Spire Image 2", s=5)
    
    pyplot.show()