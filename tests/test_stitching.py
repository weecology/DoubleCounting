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
    output_path = Path(os.path.join(test_data_dir, "output"))
    output_path.mkdir(parents=True, exist_ok=True)
    print(output_path)

    create_sfm_model(image_dir=image_dir, image_paths=image_paths, output_path=output_path)

    assert os.path.exists(output_path / "features.h5")
    assert os.path.exists(output_path / "matches.h5")

def test_align_and_delete(predictions, test_data_dir, image_dir, image_paths):
    output_path = Path(os.path.join(test_data_dir, "output"))
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

def test_transform_2d_to_3d(image_dir, image_paths, test_data_dir):
    # Define test inputs
    x = 100
    y = 200

    output_path = Path(os.path.join(test_data_dir, "output"))
    output_path.mkdir(parents=True, exist_ok=True)
    model = create_sfm_model(image_dir=image_dir, image_paths=image_paths, output_path=output_path)

    # Call the function
    intrinsic = model.cameras[1].params
    extrinsic = model.images[1].cam_from_world
    result = transform_2d_to_3d(x, y, intrinsic, extrinsic)

    # Check the output
    assert len(result) == 3

    # View the output in matplotlib, plot the x_centroid, y_centroid, z_centroid coordinates
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result, result[1], result[2])
    pyplot.show()