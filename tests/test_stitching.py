from scripts.stitching import create_sfm_model, align_and_delete, compute_homography_matrix, warp_points, align_predictions, remove_predictions
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
def predictions(test_data_dir):
    cache_dir = os.path.join(test_data_dir, "cache")
    cache_file = os.path.join(cache_dir, "predictions.pkl")

    image_paths = glob.glob(os.path.join(test_data_dir, "birds", "*.JPG"))
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

@pytest.fixture
def model_birds(test_data_dir):
    output_path = Path(os.path.join(test_data_dir, "output_birds"))
    output_path.mkdir(parents=True, exist_ok=True)
    image_dir = Path(os.path.join(os.path.dirname(__file__),"data/birds"))
    references = [str(p.relative_to(image_dir)) for p in (image_dir).iterdir()]
    return create_sfm_model(image_dir=image_dir, references=references, output_path=output_path)

def test_create_sfm_model(test_data_dir):
    # Call the function, create a local directory in test to save outputs
    output_path = Path(os.path.join(test_data_dir, "output_birds"))
    output_path.mkdir(parents=True, exist_ok=True)
    print(output_path)
    image_dir = Path(os.path.join(os.path.dirname(__file__),"data/birds"))
    references = [str(p.relative_to(image_dir)) for p in (image_dir).iterdir()]
    create_sfm_model(image_dir=image_dir, output_path=output_path, references=references)

    assert os.path.exists(output_path / "features.h5")
    assert os.path.exists(output_path / "matches.h5")

def test_compute_homography_matrix_and_warp(test_data_dir, model_birds):
    """Compute homography matrix and warp points from one image to another."""

    image_1 = "DSC_2522.JPG"
    image_2 = "DSC_2523.JPG"

    # Coordinates of the birds
    bird1_image_1_x = 3043
    bird1_image_1_y = 2552

    bird1_image_2_x = 3003
    bird1_image_2_y = 2843

    bird2_image_1_x = 4113
    bird2_image_1_y = 1398

    bird2_image_2_x = 3780
    bird2_image_2_y = 1626

    image_dir = Path(os.path.join(os.path.dirname(__file__),"data/birds"))

    # Plot keypoints of top and left spire
    keypoints = [
        np.array([[bird1_image_1_x, bird1_image_1_y], [bird2_image_1_x, bird2_image_1_y]]),
        np.array([[bird1_image_2_x, bird1_image_2_y], [bird2_image_2_x, bird2_image_2_y]])
    ]
    plot_images([read_image(image_dir / x) for x in [image_1, image_2]], dpi=200)
    plot_keypoints(keypoints,colors=['purple','orange'])

    # Call the function with the proper index
    camera = model_birds.cameras[1]
    
    output_path = Path(os.path.join(test_data_dir, "output_birds"))

    answer = compute_homography_matrix(
        model=model_birds,
        h5_file=output_path / "matches.h5",
        image1_name=image_1,
        image2_name=image_2)

    # Warp the keypoints points
    keypoints = [np.array([[bird1_image_1_x, bird1_image_1_y], [bird2_image_1_x, bird2_image_1_y]]), np.array([[bird1_image_2_x, bird1_image_2_y], [bird2_image_2_x, bird2_image_2_y]])]
    warped_points = warp_points(answer["H"], keypoints[0])

    # Warped points overlayed on image 2
    pyplot.figure()
    image = read_image(image_dir / image_2)
    # Plot the warped points
    for point in warped_points:
        pyplot.scatter(point[0][0], point[0][1], c='purple', marker='o', alpha=0.75, s=5)
    # Plot the original points
    for point in keypoints[1]:
        pyplot.scatter(point[0], point[1], c='orange', marker='o', alpha=0.75, s=5)
    # Show the plot
    pyplot.imshow(image)
    pyplot.title("Predicted location of image 1 annotations on image 2")

    # Show the plot
    pyplot.show()

    # Assert that keypoints are in the same location
    assert np.allclose(warped_points[0], keypoints[1][0], atol=30)

def test_align_predictions(predictions, test_data_dir, model_birds):
    """Test whether we can take predictions from one image and overlay on another as boxes"""
    image_dir = Path(os.path.join(os.path.dirname(__file__),"data/birds"))
    output_path = Path(os.path.join(test_data_dir, "output_birds"))

    # Compute homography   
    src_image_name = predictions.image_path.unique()[0]
    dst_image_name = predictions.image_path.unique()[1]
    homography = compute_homography_matrix(model=model_birds, h5_file=output_path / "matches.h5", image1_name=src_image_name, image2_name=dst_image_name)
    src_image_predictions = predictions[predictions.image_path == src_image_name]
    
    # Transform the predictions
    aligned_predictions = align_predictions(predictions=src_image_predictions, homography_matrix=homography["H"])

    # Same number of predictions
    assert len(aligned_predictions) == len(src_image_predictions)

    # View boxes on dst image
    image = cv2.imread(os.path.join(image_dir.__str__(), dst_image_name))
    for index, row in aligned_predictions.iterrows():
        x = row["xmin"]
        y = row["ymin"]
        w = row["xmax"] - row["xmin"]
        h = row["ymax"] - row["ymin"]
        # Draw bounding box
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 165, 255), 4)

    # Add dst predictions
    dst_image_predictions = predictions[predictions.image_path == dst_image_name]
    for index, row in dst_image_predictions.iterrows():
        x = row["xmin"]
        y = row["ymin"]
        w = row["xmax"] - row["xmin"]
        h = row["ymax"] - row["ymin"]
        # Draw bounding box
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 4)

    pyplot.imshow(image[:,:,::-1])
    pyplot.title("Aligned source and destination predictions")
    pyplot.show()

    # Delete predictions and show the result
    predictions_to_remove = pd.concat([aligned_predictions, predictions[predictions.image_path == dst_image_name]])
    remaining_predictions = remove_predictions(predictions_to_remove)

    # Final predictions
    pyplot.figure()
    final_image = cv2.imread(os.path.join(image_dir.__str__(), dst_image_name))
    for index, row in remaining_predictions.iterrows():
        x = row["xmin"]
        y = row["ymin"]
        w = row["xmax"] - row["xmin"]
        h = row["ymax"] - row["ymin"]
        # Draw bounding box
        cv2.rectangle(final_image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 192, 203), 5)

    pyplot.imshow(final_image[:,:,::-1])
    pyplot.title("final predictions")
    pyplot.show()


@pytest.mark.parametrize("strategy", ["left-hand", "highest-score"])
def test_align_and_delete(predictions, test_data_dir, model_birds, strategy):
    image_dir = Path(os.path.join(os.path.dirname(__file__),"data/birds"))
    output_path = Path(os.path.join(test_data_dir, "output_birds"))
    images_to_plot = predictions["image_path"].unique()
    images_to_plot.sort()
    
    fig, axs = pyplot.subplots(3, 2, figsize=(10, 10))
    axs = axs.flatten()

    # for i, image_path in enumerate(images_to_plot):
    #     image = cv2.imread(os.path.join(image_dir.__str__(), image_path))
    #     original_predictions = predictions[predictions["image_path"] == image_path]
    #     for index, row in original_predictions.iterrows():
    #         x = row["xmin"]
    #         y = row["ymin"]
    #         w = row["xmax"] - row["xmin"]
    #         h = row["ymax"] - row["ymin"]
    #         # Draw bounding box
    #         cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 5)
    #     axs[i].imshow(image[:,:,::-1])
    #     axs[i].set_title(f"Original predictions for {image_path}")
    # pyplot.tight_layout()
    # pyplot.show()

    final_predictions = align_and_delete(predictions=predictions, model=model_birds, image_dir=image_dir, matching_h5_file=output_path / "matches.h5", strategy=strategy)

    # For each image in predictions, plot predictions using geopandas and use the image_path as the background using rasterio
    fig, axs = pyplot.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    images_to_plot = final_predictions["image_path"].unique()
    images_to_plot.sort()
    for i, image_path in enumerate(images_to_plot):
        image = cv2.imread(os.path.join(image_dir.__str__(), image_path))
        # Show originals as well
        original_predictions = predictions[predictions["image_path"] == image_path]
        for index, row in original_predictions.iterrows():
            x = row["xmin"]
            y = row["ymin"]
            w = row["xmax"] - row["xmin"]
            h = row["ymax"] - row["ymin"]
            # Draw bounding box
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 7)
        unique_predictions_image = final_predictions[final_predictions["image_path"] == image_path]
        for index, row in unique_predictions_image.iterrows():
            x = row["xmin"]
            y = row["ymin"]
            w = row["xmax"] - row["xmin"]
            h = row["ymax"] - row["ymin"]
            # Draw bounding box
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (182, 192, 255), 5)
        axs[i].imshow(image[:,:,::-1])
        axs[i].set_title(f"Final predictions for {image_path}")

    pyplot.tight_layout()
    pyplot.show()

    print("Tests passed!")