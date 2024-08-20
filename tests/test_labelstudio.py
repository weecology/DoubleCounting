#Test_labelstudio.py
from scripts.labelstudio import upload
from scripts.predict import predict
import os
import pytest

@pytest.mark.skipif("label_studio_api_key.txt" not in os.listdir("/blue/ewhite/everglades/label_studio/"), reason="No API key found")
def test_upload_to_labelstudio():        
    # Set the Label studio API key as env variable
    with open("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt", "r") as file:
        api_key = file.read().strip()
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    images = ["tests/data/DSC_2520.JPG", "tests/data/DSC_2521.JPG"]
    model_path = "/blue/ewhite/everglades/Zooniverse/20220910_182547/species_model.pl"
    save_dir = "/blue/ewhite/everglades/Airplane/annotations"
    predictions = []
    
    if os.path.exists(model_path):
        image_paths = predict(image_dir="/blue/ewhite/everglades/Airplane/images_to_predict", save_dir=save_dir, model_path=model_path)
        basename = os.path.splitext(os.path.basename(image_paths[0]))[0]
        csv_path = os.path.join(save_dir, "{}.csv".format(basename))
        predictions.append(csv_path)  
    
    upload(
            user="ben",
            host="serenity.ifas.ufl.edu",
            key_filename="/home/b.weinstein/.ssh/id_rsa.pub",
            label_studio_url="https://labelstudio.naturecast.org/",
            images=images,
            preannotations=predictions,
            folder_name="/pgsql/retrieverdash/everglades-label-studio/everglades-data"
        )