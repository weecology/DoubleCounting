# Predict module for labeling pipeline
from deepforest import main
import os
import glob

def predict(model_path, save_dir=None, image_dir=None, image_paths=None):
    """
    Predict bounding boxes for objects in images using a trained model. Can either specify a directory or a list of image paths

    Args:
        image_dir (str): Directory path containing the input images.
        save_dir (str): Directory path to save the predicted bounding box results.
        model_path (str): Path to the trained model checkpoint.
        image_paths (list, optional): List of specific image paths to predict bounding boxes for.
            If not provided, all images in the `image_dir` will be used.

    Returns:
        None
    """
    if image_paths is None:
        image_paths = glob.glob("{}/*.JPG".format(image_dir))
    if model_path is None:
        m = main.deepforest()
        m.use_bird_release()
    else:
        m = main.deepforest.load_from_checkpoint(model_path)
    
    predictions = []
    for image_path in image_paths:
        boxes = m.predict_tile(image_path, patch_size=1500, patch_overlap=0.05)
        boxes = boxes[boxes["score"] > 0.2]
        if save_dir:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            filename = os.path.join(save_dir, "{}.csv".format(basename))
            boxes.to_csv(filename)
            predictions.append(filename)
        else:
            predictions.append(boxes)

    return image_paths, predictions