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

if __name__ == "__main__":
    import argparse
    from scripts.labelstudio import upload
    parser = argparse.ArgumentParser(description='Predict bounding boxes on an image')
    parser.add_argument('--save_dir', type=str, help='Directory to save csv and svg files')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--user', type=str, help='The username for the SSH connection')
    parser.add_argument('--host', type=str, help='The hostname or IP address of the remote server')
    parser.add_argument('--image_dir', type=str, help='Directory containing the images to predict')
    parser.add_argument('--key_filename', type=str, help='The path to the private key file for authentication')
    parser.add_argument('--label_studio_url', type=str, help='The URL of the Label Studio server')
    parser.add_argument('--label_studio_project', type=str, help='The name of the Label Studio project')
    parser.add_argument("--label_studio_folder", type=str, help="The name of the folder on the remote server where the images will be uploaded")

    args = parser.parse_args()

    # Set the Label studio API key as env variable
    with open("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt", "r") as file:
        api_key = file.read().strip()
    os.environ["LABEL_STUDIO_API_KEY"] = api_key
    args = parser.parse_args()
    images, predictions = predict(args.image_dir, args.save_dir, args.model_path)

    upload(
        user=args.user,
        host=args.host,
        key_filename=args.key_filename,
        label_studio_url=args.label_studio_url,
        label_studio_project=args.label_studio_project,
        images=images,
        preannotations=predictions,
        label_studio_folder=args.label_studio_folder
    )

