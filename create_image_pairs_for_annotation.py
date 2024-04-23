# Create label-studio json for pair on images right img-left, img-right tags
from sympy import im
from scripts.labelstudio import upload_paired_images
import argparse
import os

parser = argparse.ArgumentParser(description='Create image pairs for annotation')
parser.add_argument('--img_left', type=str, help='Path to left image')
parser.add_argument('--img_right', type=str, help='Path to right image')
parser.add_argument('--user', type=str, help='SFTP username')
parser.add_argument('--folder', type=str, help='SFTP folder')
parser.add_argument('--host', type=str, help='SFTP host')
parser.add_argument('--key_filename', type=str, help='Path to private key')
parser.add_argument('--label_studio_url', type=str, help='Label Studio URL')
parser.add_argument('--label_studio_project', type=str, help='Label Studio project name')
parser.add_argument('--label_studio_folder', type=str, help='Label Studio folder name')
parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
args = parser.parse_args()

# Set the Label studio API key as env variable
with open("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt", "r") as file:
    api_key = file.read().strip()
os.environ["LABEL_STUDIO_API_KEY"] = api_key

upload_paired_images(
    img_list=[args.img_left, args.img_right],
    user=args.user,
    folder=args.folder,
    host=args.host,
    key_filename=args.key_filename,
    label_studio_url=args.label_studio_url,
    label_studio_project=args.label_studio_project,
    label_studio_folder=args.label_studio_folder,
    model_path=args.model_path
)

