import subprocess
import argparse

def look_for_images(source_path, destination_path):
    try:
        subprocess.run(["rclone", "copy", source_path, destination_path], check=True)
        print("Images copied successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Source path for images")
    parser.add_argument("--destination", help="Destination path for images")
    args = parser.parse_args()

    look_for_images(args.source, args.destination)