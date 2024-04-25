from scripts.stitching import create_sfm_model, align_and_delete, transform_2d_to_3d
from scripts.predict import predict
import cv2
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

@pytest.fixture()
def image_paths(test_data_dir):
    image_paths = glob.glob(os.path.join(test_data_dir, "*.JPG"))
    image_paths.sort()
    return image_paths

@pytest.fixture()
def image_dir(test_data_dir):
    return Path(test_data_dir)

@pytest.fixture()
def predictions(image_paths, test_data_dir):
    cache_dir = os.path.join(test_data_dir, "cache")
    cache_file = os.path.join(cache_dir, "predictions.pkl")

    if os.path.exists(cache_file):
        # Load predictions from cache
        predictions = pd.read_pickle(cache_file)
    else:
        # Generate predictions and save to cache
        _, predictions = predict(image_paths=image_paths, save_dir=None, model_path=None)
        predictions = pd.concat(predictions)
        os.makedirs(cache_dir, exist_ok=True)
        predictions.to_pickle(cache_file)

    return predictions

def test_create_sfm_model(image_dir, image_paths, test_data_dir):
    # Call the function, create a local directory in test to save outputs
    output_path = Path(os.path.join(test_data_dir, "output_birds"))
    output_path.mkdir(parents=True, exist_ok=True)
    print(output_path)

    create_sfm_model(image_dir=image_dir, image_paths=image_paths, output_path=output_path)

    assert os.path.exists(output_path / "features.h5")
    assert os.path.exists(output_path / "matches.h5")

def test_align_and_delete(predictions, test_data_dir, image_dir, image_paths):
    output_path = Path(os.path.join(test_data_dir, "output_birds"))
    output_path.mkdir(parents=True, exist_ok=True)    
    model = create_sfm_model(image_dir=image_dir, image_paths=image_paths, output_path=output_path)
    unique_predictions = align_and_delete(predictions=predictions, model=model, image_dir=image_dir, visualize=True)

    # For each image in predictions, plot predictions using geopandas and use the image_path as the background using rasterio
    for image_path in unique_predictions["image_path"]:
        image = cv2.imread(os.path.join(image_dir.__str__(), image_path))
        unique_predictions_image = unique_predictions[unique_predictions["image_path"] == image_path]
        for index, row in unique_predictions_image.iterrows():
            x = row["xmin"]
            y = row["ymin"]
            w = row["xmax"] - row["xmin"]
            h = row["ymax"] - row["ymin"]
            # Draw bounding box
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        pyplot.imshow(image)
        pyplot.show()

def test_transform_2d_to_3d_simple(image_dir, test_data_dir):
    # Create a test case for a simple images. Point to the top of the spire in the church image and get the 3D coordinates in two images, they should overlap

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

    model = create_sfm_model(image_dir=image_dir, references=references, output_path=output_path)

    # Call the function with the proper index
    image_1 = model.find_image_with_name("simple1.jpg")
    intrinsic_1 = model.cameras[2].params
    extrinsic_1 = image_1.cam_from_world

    image_2 = model.find_image_with_name("simple2.jpg")
    intrinsic_2 = model.cameras[1].params
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

def test_transform_2d_to_3d_birds(image_dir, test_data_dir):
    # Create a test case for a simple images. Point to the top of the spire in the church image and get the 3D coordinates in two images, they should overlap

    image_1 = "DSC_2522.JPG"
    image_2 = "DSC_2523.JPG"

    # Coordinates of the 
    bird1_image_1_x = 3043
    bird1_image_1_y = 2552

    bird1_image_2_x = 3003
    bird1_image_2_y = 2843

    bird2_image_1_x = 4113
    bird2_image_1_y = 1398

    bird2_image_2_x = 3780
    bird2_image_2_y = 1626


    image_dir = Path(os.path.join(os.path.dirname(__file__),"data/birds"))
    output_path = Path(os.path.join(test_data_dir, "output_birds"))
    output_path.mkdir(parents=True, exist_ok=True)
    references = [str(p.relative_to(image_dir)) for p in (image_dir).iterdir()]

    # List and view images
    images = [read_image(image_dir / x) for x in references]
    plot_images(images, dpi=200)
    # Plot keypoints of top and left spire
    keypoints = [np.array([[bird1_image_1_x, bird1_image_1_y], [bird2_image_1_x, bird2_image_1_y]]), np.array([[bird1_image_2_x, bird1_image_2_y], [bird2_image_2_x, bird2_image_2_y]])]
    plot_keypoints(keypoints)

    model = create_sfm_model(image_dir=image_dir, references=references, output_path=output_path, )

    # Call the function with the proper index
    image_1 = model.find_image_with_name(image_1)
    intrinsic_1 = model.cameras[2].params
    extrinsic_1 = image_1.cam_from_world

    image_2 = model.find_image_with_name(image_2)
    intrinsic_2 = model.cameras[1].params
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

    # Check if the 3D coordinates are close
    np.testing.assert_allclose(result_1, result_2, rtol=1e-3)
