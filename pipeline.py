# Read the images from the image_dir, predict the bounding boxes and upload the images to the Label Studio server

import argparse
from scripts.labelstudio import upload
from scripts.predict import predict
from scripts.stitching import align_and_delete, create_sfm_model
import os
from datetime import datetime
from pathlib import Path

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
    
def wrapper(folder_path, args):
    images, predictions = predict(folder_path, args.save_dir, args.model_path)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    basename = os.path.basename(folder_path)
    output_path = Path(os.path.join(args.save_dir, basename,timestamp))
    output_path.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir)
    references = [str(p.relative_to(image_dir)) for p in (image_dir).iterdir()]

    sfm_model = create_sfm_model(image_dir=image_dir, output_path=output_path, references=references)
    filtered_predictions = align_and_delete(predictions=predictions, model=sfm_model, image_dir=image_dir, matching_h5_file=output_path / "matches.h5", strategy="left-hand")

    upload(
        user=args.user,
        host=args.host,
        key_filename=args.key_filename,
        label_studio_url=args.label_studio_url,
        images=images,
        preannotations=filtered_predictions,
        label_studio_folder=args.label_studio_folder
    )

# for each folder in image_dir, get the folder name and run the wrapper function
for folder in os.listdir(args.image_dir):
    folder_path = os.path.join(args.image_dir, folder)
    wrapper(folder_path, args)
