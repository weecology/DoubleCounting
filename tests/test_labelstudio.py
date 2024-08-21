#Test_labelstudio.py
from scripts.labelstudio import upload
from scripts.predict import predict
import os
import pytest

@pytest.mark.skipif("label_studio_api_key.txt" not in os.listdir("tests") , reason="No API key found")
def test_upload_to_labelstudio(tmpdir):        
    # Set the Label studio API key as env variable
    with open("tests/label_studio_api_key.txt", "r") as file:
        api_key = file.read().strip()
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    images = ["tests/data/birds/DSC_2520.JPG", "tests/data/birds/DSC_2521.JPG"]
    save_dir = tmpdir
    predictions = []
    
    image_paths, predictions = predict(image_paths=images, save_dir=save_dir, model_path=None)
    basename = os.path.splitext(os.path.basename(image_paths[0]))[0]
    csv_path = os.path.join(save_dir, "{}.csv".format(basename))
    predictions.append(csv_path)  
    
    upload(
            user="ben",
            host="serenity.ifas.ufl.edu",
            key_filename="/home/b.weinstein/.ssh/id_rsa",
            label_studio_url="https://labelstudio.naturecast.org/",
            images=images,
            preannotations=predictions,
            folder_name="/pgsql/retrieverdash/everglades-label-studio/everglades-data"
        )