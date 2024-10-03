
from pyexpat import model
import paramiko
import os
import datetime
import pandas as pd
from label_studio_sdk import Client
from PIL import Image
from scripts.predict import predict

def connect_to_label_studio(url, project_name, label_config=None):
    """Connect to the Label Studio server.
    Args:
        port (int, optional): The port of the Label Studio server. Defaults to 8080.
        host (str, optional): The host of the Label Studio server. Defaults to "localhost". 
    Returns:
        str: The URL of the Label Studio server.
    """
    ls = Client(url=url, api_key=os.environ["LABEL_STUDIO_API_KEY"])
    ls.check_connection()

    if label_config:
        project = ls.start_project(title=project_name, label_config=label_config)
    else:
        projects = ls.list_projects()
        project = [x for x in projects if x.get_params()["title"] == project_name][0]

    return project

def create_client(user, host, key_filename):
    """
    Create an SFTP client to download annotations from Label Studio.

    Args:
        user (str): The username for the SSH connection.
        host (str): The hostname or IP address of the remote server.
        key_filename (str): The path to the private key file for authentication.

    Returns:
        paramiko.SFTPClient: The SFTP client object.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, key_filename=key_filename)
    sftp = ssh.open_sftp()

    return sftp

def delete_completed_tasks(label_studio_project):
    """
    Deletes the completed tasks from the given Label Studio project.

    Args:
        label_studio_project: The Label Studio project object.

    Returns:
        None
    """
    tasks = label_studio_project.get_labeled_tasks()
    for task in tasks:
        label_studio_project.delete_task(task["id"])

def import_image_tasks(label_studio_project, image_names, local_image_dir, predictions=None):
    """
    Imports image tasks into a Label Studio project.

    Args:
        label_studio_project (LabelStudioProject): The Label Studio project to import tasks into.
        image_names (list): A list of image names to import as tasks.
        local_image_dir (str): The local directory where the images are stored.
        predictions (dict, optional): A dictionary containing the predictions for the images. Key is image name e.g. 'DSC_2577.JPG'

    Returns:
        None
    """
    df = pd.concat([value for key,value in predictions.items()])
    unique_images = df.image_path.unique()
    data_dict = {}
    for i, image_name in enumerate(unique_images):
        data_dict["img{}".format(i+1)] = os.path.join("/data/local-files/?d=input/", os.path.basename(image_name))    
    
    image_json = label_studio_bbox_format(local_image_dir, df)

    upload_dict = {"data": data_dict, "predictions": [image_json]}
    label_studio_project.import_tasks(upload_dict)

def download_completed_tasks(label_studio_project, train_csv_folder):
    """
    Downloads completed tasks from a Label Studio project and saves them as a CSV file.

    Args:
        label_studio_project (LabelStudioProject): The Label Studio project object.
        train_csv_folder (str): The folder path where the CSV file will be saved.

    Returns:
        pandas.DataFrame: The downloaded annotations as a pandas DataFrame.
    """
    labeled_tasks = label_studio_project.get_labeled_tasks()
    if not labeled_tasks:
        print("No new annotations")
        return None
    else:
        images, labels = [], []
    for labeled_task in labeled_tasks:
        image_path = os.path.basename(labeled_task['data']['image'])
        images.append(image_path)
        label_json = labeled_task['annotations'][0]["result"]
        if len(label_json) == 0:
            result = {
                    "image_path": image_path,
                    "xmin": None,
                    "ymin": None,
                    "xmax": None,
                    "ymax": None,
                    "label": None,
                    "annotator":labeled_task["annotations"][0]["created_username"]
                }
            result = pd.DataFrame(result, index=[0])
        else:
            result = convert_json_to_dataframe(label_json, image_path)
            result["annotator"] = labeled_task["annotations"][0]["created_username"]
        labels.append(result)

    annotations =  pd.concat(labels) 
    print("There are {} new annotations".format(annotations.shape[0]))
    annotations = annotations[~(annotations.label=="Help me!")]
    annotations.loc[annotations.label=="Unidentified White","label"] = "Unknown White"

    # Save csv in dir with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_path = os.path.join(train_csv_folder, "train_{}.csv".format(timestamp))
    annotations.to_csv(train_path, index=False)

    return annotations

def upload_images(sftp_client, images, folder_name):
    """
    Uploads a list of images to a remote server using SFTP.

    Args:
        sftp_client (SFTPClient): An instance of the SFTPClient class for SFTP file transfer.
        images (list): A list of image file paths to be uploaded.
        folder_name (str): The name of the folder on the remote server where the images will be uploaded.

    Returns:
        None
    """
    # SCP file transfer
    for image in images:
        sftp_client.put(image, os.path.join(folder_name,"input",os.path.basename(image)))
        print(f"Uploaded {image} successfully")

def remove_annotated_images_remote_server(sftp_client, annotations, folder_name):
    """Remove images that have been annotated on the Label Studio server.

    Args:
        sftp_client (SFTPClient): The SFTP client used to connect to the remote server.
        annotations (DataFrame): A DataFrame containing the annotations for the images.
        folder_name (str): The name of the folder where the images and annotations are stored.
    """
    # Delete images using SSH
    for image in annotations.image_path.unique():
        remote_path = os.path.join(folder_name, "input", os.path.basename(image))
        # Archive annotations using SSH
        archive_annotation_path = os.path.join(folder_name, "archive", os.path.basename(image))
        # sftp check if dir exists
        try:
            sftp_client.listdir(os.path.join(folder_name, "archive"))
        except FileNotFoundError:
            raise FileNotFoundError("The archive directory {} does not exist.".format(os.path.join(folder_name, "archive")))
        
        sftp_client.rename(remote_path, archive_annotation_path)
        print(f"Archived {image} successfully")

def label_studio_bbox_format(local_image_dir, preannotations, from_name="label"):
    """
    Create a JSON string for a single image in the Label Studio API format.

    Args:
        local_image_dir (str): The local directory where the images are stored.
        preannotations (DataFrame): A DataFrame containing the preannotations for the image.
        to_name (str, optional): The name of the image. Defaults to "image".
        from_name (str, optional): The name of the label. Defaults to "label".

    Returns:
        dict: The JSON string in the Label Studio API format.
    """
    predictions = []
    image_path = preannotations.image_path.unique()[0]
    original_width, original_height = Image.open(os.path.join(local_image_dir, os.path.basename(image_path))).size

    #Unique images and their index
    unique_images = preannotations.image_path.unique()
    image_index_dict = {}
    for index, image in enumerate(unique_images):
        image_index_dict[image] = "img{}".format(index+1)

    for index, row in preannotations.iterrows():
        result = {
            "value": {
                "x": row['xmin'] / original_width * 100,
                "y": row['ymin'] / original_height * 100,
                "width": (row['xmax'] - row['xmin']) / original_width * 100,
                "height": (row['ymax'] - row['ymin']) / original_height * 100,
                "rotation": 0,
                "rectanglelabels": [row["label"]]
            },
            "score": row["score"],
            "to_name": image_index_dict[row["image_path"]],
            "type": "rectanglelabels",
            "from_name": image_index_dict[row["image_path"]].replace("img", "label"),
            "original_width": original_width,   
            "original_height": original_height
        }
        predictions.append(result)
            
    return {"result": predictions}

def convert_json_to_dataframe(x, image_path):
    # Loop through annotations and convert to pandas DataFrame
    results = []
    for annotation in x:
        xmin = annotation["value"]["x"] / 100 * annotation["original_width"]
        ymin = annotation["value"]["y"] / 100 * annotation["original_height"]
        xmax = (annotation["value"]["width"] / 100 + annotation["value"]["x"] / 100) * annotation["original_width"]
        ymax = (annotation["value"]["height"] / 100 + annotation["value"]["y"] / 100) * annotation["original_height"]
        label = annotation["value"]["rectanglelabels"][0]

        # Create dictionary
        result = {
            "image_path": image_path,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "label": label,
        }

        # Append to list
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df

def create_label_config(predictions):
    """
    Creates a Label Studio XML configuration based on the provided predictions.

    Args:
        predictions (dict): A dictionary containing the preannotations for each image.

    Returns:
        str: The Label Studio XML configuration.
    """

    xml = '''<View>
        <Header value="Select unique birds in each image to create a full colony count" />'''

    for i, image_name in enumerate(predictions.keys()):
        xml += f'''
            <View style="display: flex;">
                <View>
                <Image name="img{i+1}" value="$img{i+1}"/>
                <RectangleLabels name="label{i+1}" toName="img{i+1}">
                    <Label value="Great Egret" background="#FFA39E"/>
                    <Label value="Great Blue Heron" background="#D4380D"/>
                    <Label value="Wood Stork" background="#FFC069"/>
                    <Label value="Snowy Egret" background="#AD8B00"/>
                    <Label value="Anhinga" background="#D3F261"/>
                    <Label value="Unidentified White" background="#389E0D"/>
                    <Label value="White Ibis" background="#5CDBD3"/>
                    <Label value="Nest" background="#FFA39E"/>
                    <Label value="Help me!" background="#D4380D"/>
                    <Label value="Eggs" background="#FFA39E"/>
                    <Label value="Roseate Spoonbill" background="#FFA39E"/>
                </RectangleLabels>
                </View>
            </View>'''

    xml += '''
    </View>'''

    return xml

def upload(user, host, key_filename, label_studio_url, images, preannotations, keypoints, folder_name):
    """
    Uploads images to Label Studio and imports image tasks with preannotations.

    Args:
        user (str): The username for the SFTP connection.
        host (str): The hostname for the SFTP connection.
        key_filename (str): The path to the private key file for the SFTP connection.
        label_studio_url (str): The URL of the Label Studio instance.
        images (str): List of paths to the images to upload. Assumes that all images are in the same directory!
        preannotations (str): The csv files containing the preannotations.
        folder_name (str): The name of the folder on the remote server where the images will be uploaded.
        keypoints ()

    Returns:
        None
    """
    #Read each csv file and create a dictionary with the image name as the key and the dataframe as the value
    preannotations = {os.path.splitext(os.path.basename(preannotation))[0]: pd.read_csv(preannotation) for preannotation in preannotations}
    sftp_client = create_client(user=user, host=host, key_filename=key_filename)
    label_config = create_label_config(predictions=preannotations)
    project_name = os.path.basename(os.path.dirname(images[0]))
    label_studio_project = connect_to_label_studio(url=label_studio_url, project_name=project_name, label_config=label_config)
    upload_images(sftp_client=sftp_client, images=images, folder_name=folder_name)
    import_image_tasks(label_studio_project=label_studio_project, image_names=images, local_image_dir=os.path.dirname(images[0]), predictions=preannotations)
    import_keypoints(label_studio_project=label_studio_project, keypoints=keypoints)


def import_keypoints(keypoints, label_studio_project):
    """
    Imports keypoints from a CSV file into a Label Studio project.

    Args:
        keypoint_csv_file (str): The path to the CSV file containing the keypoints.
        label_studio_project (LabelStudioProject): The Label Studio project to import the keypoints into.

    Returns:
        None
    """
    data_dict = {}
    image_name_order = keypoints["image_path"].drop_duplicates().reset_index(drop=True, inplace=True)
    image_name_order.index = keypoints.index + 1
    image_name_order = image_name_order.to_dict()
    image_name_order = {v: k for k, v in image_name_order.items()}

    keypoints_data = []
    for i, row in keypoints.iterrows():
        data_dict["img{}".format(i+1)] = os.path.join("/data/local-files/?d=input/", os.path.basename(row["image_path"]))
        image = image_name_order[row["image_path"]]
        x = row["x"]
        y = row["y"]
        color = row["color"]
        keypoint_json = label_studio_keypoint_format(image, x, y, color)
        keypoints_data.append(keypoint_json)

    upload_dict = {"data": data_dict, "predictions": keypoints_data}
    label_studio_project.import_tasks(upload_dict)

def label_studio_keypoint_format(image, x, y, color):
    """
    Create a JSON string for a single keypoint in the Label Studio API format.

    Args:io
        image (str): The name of the image.
        x (float): The x-coordinate of the keypoint.
        y (float): The y-coordinate of the keypoint.
        color (str): The color of the keypoint.

    Returns:
        dict: The JSON string in the Label Studio API format.
    """
    result = {
        "value": {
            "x": x,
            "y": y,
            "width": 0.01,  # width and height are small since it's a keypoint
            "height": 0.01,
            "rotation": 0,
            'fillColor': color
        },
        "to_name": image,  # assuming the keypoint is for the first image
        "type": "KeyPoint",
        "from_name": "keypoint1"
    }
    return {"result": [result]}