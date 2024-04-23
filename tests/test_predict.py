import os
from scripts.predict import predict
from deepforest import get_data

def test_predict(tmpdir):
    image_paths = [get_data('OSBS_029.png')]
    save_dir = tmpdir.strpath
    predict(image_paths, save_dir, model_path=None)
    basename = os.path.splitext(os.path.basename(image_paths[0]))[0]
    csv_path = os.path.join(save_dir, "{}.csv".format(basename))
    
    # Check if the .csv file is created
    assert os.path.exists(csv_path)

def test_predict_pipeline():
    # Only test if no HPC
    model_path = "/blue/ewhite/everglades/Zooniverse/20220910_182547/species_model.pl"
    save_dir = "/blue/ewhite/everglades/Airplane/annotations"
    if os.path.exists(model_path):
        image_paths = predict(image_dir="/blue/ewhite/everglades/Airplane/images_to_predict", save_dir=save_dir, model_path=model_path)
        basename = os.path.splitext(os.path.basename(image_paths[0]))[0]
        csv_path = os.path.join(save_dir, "{}.csv".format(basename))
        
        # Check if the .csv file is created
        assert os.path.exists(csv_path)
