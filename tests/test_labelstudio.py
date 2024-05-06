#Test_labelstudio.py
from scripts.labelstudio import upload
import os
import pytest

@pytest.mark.skipif("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt" not in os.listdir("/blue/ewhite/everglades/label_studio/"), reason="No API key found")
def test_upload_to_labelstudio():        
    # Set the Label studio API key as env variable
    with open("/blue/ewhite/everglades/label_studio/label_studio_api_key.txt", "r") as file:
        api_key = file.read().strip()
    os.environ["LABEL_STUDIO_API_KEY"] = api_key

    images = ["/blue/ewhite/everglades/Airplane/images_to_predict/DSC_3847.JPG"]
    predictions = ["/blue/ewhite/everglades/Airplane/predictions/DSC_3847.csv"]
    upload(
            user="ben",
            host="serenity.ifas.ufl.edu",
            key_filename="/home/b.weinstein/.ssh/id_rsa.pub",
            label_studio_url="https://labelstudio.naturecast.org/",
            label_studio_project="Airplane Photos",
            images=images,
            preannotations=predictions,
            label_studio_folder="/pgsql/retrieverdash/everglades-label-studio/everglades-data"
        )