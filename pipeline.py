# Read the images from the image_dir, predict the bounding boxes and upload the images to the Label Studio server

import argparse
from scripts.labelstudio import upload
from scripts.predict import predict
from scripts.stitching import align_and_delete, create_sfm_model
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description='Predict bounding boxes on an image')
parser.add_argument('--save_dir', type=str, help='Directory to save csv and svg files')
parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
parser.add_argument('--user', type=str, help='The username for the SSH connection')
parser.add_argument('--host', type=str, help='The hostname or IP address of the remote server')
parser.add_argument('--image_dir', type=str, help='Directory containing the images to predict')
parser.add_argument('--key_filename', type=str, help='The path to the private key file for authentication')
parser.add_argument('--label_studio_url', type=str, help='The URL of the Label Studio server')
parser.add_argument("--label_studio_folder", type=str, help="The name of the folder on the remote server where the images will be uploaded")

args = parser.parse_args()

# Set the Label studio API key as env variable
with open("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt", "r") as file:
    api_key = file.read().strip()
os.environ["LABEL_STUDIO_API_KEY"] = api_key
args = parser.parse_args()
    
def wrapper(folder_path, args):
    images, predictions_csvs = predict(image_dir=folder_path, save_dir=args.save_dir, model_path=args.model_path)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    basename = os.path.basename(folder_path)
    output_path = Path(os.path.join(args.save_dir, basename,timestamp))
    output_path.mkdir(parents=True, exist_ok=True)
    image_dir = Path(folder_path)
    references = [str(p.relative_to(image_dir)) for p in (image_dir).iterdir()]

    sfm_model = create_sfm_model(image_dir=image_dir, output_path=output_path, references=references)

    predictions = pd.concat([pd.read_csv(p) for p in predictions_csvs])
    filtered_predictions = align_and_delete(predictions=predictions, model=sfm_model, image_dir=image_dir, matching_h5_file=output_path / "matches.h5", strategy="left-hand")

    # Upload takes lists of image paths and prediction_csvs
    # Split filtered_predictions into a list by image_path
    split_predictions = filtered_predictions.groupby('image_path')

    # Save each group as a separate CSV file
    csv_files = []
    for image_path, group in split_predictions:
        csv_file = os.path.join(args.save_dir, basename, timestamp, f"{image_path}.csv")
        group.to_csv(csv_file, index=False)
        csv_files.append(csv_file)

    # Create a list of image paths
    image_paths = [os.path.join(folder_path, path) for path in split_predictions.groups.keys()]

    # Upload the images and CSV files to the Label Studio server
    upload(
        user=args.user,
        host=args.host,
        key_filename=args.key_filename,
        label_studio_url=args.label_studio_url,
        images=image_paths,
        preannotations=csv_files,
        folder_name=args.label_studio_folder
    )

# for each folder in image_dir, get the folder name and run the wrapper function
for folder in os.listdir(args.image_dir):
    folder_path = os.path.join(args.image_dir, folder)
    wrapper(folder_path, args)
