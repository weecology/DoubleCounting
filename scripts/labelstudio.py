from pyexpat import model
import paramiko
import os
import datetime
import pandas as pd
from label_studio_sdk import Client
from PIL import Image
import cv2
import numpy as np
import random

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

def create_label_config(predictions):
    """
    Creates a Label Studio XML configuration based on the provided predictions.

    Args:
        predictions (dict): A dictionary containing the preannotations for each image.

    Returns:
        str: The Label Studio XML configuration.
    """

    xml = '''<View>
        <Header value="Select unique birds in each image to create a full colony count" />
    '''

    for i, image_name in enumerate(predictions.keys()):
        print(i)
        if i % 2 == 0:
            xml += '<View style="display: flex;">'
        xml += f'''
                <View style="flex: 50%">
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
            '''
        if i % 2 == 1 or i == len(predictions.keys())-1:
            xml += '</View>'
    xml += '''
    </View>'''

    return xml

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

def label_studio_keypoint_format(local_image_dir, preannotations):
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
    image_path = preannotations.image.unique()[0]
    original_width, original_height = Image.open(os.path.join(local_image_dir, os.path.basename(image_path))).size

    #Unique images and their index
    unique_images = preannotations.image.unique()
    image_index_dict = {}
    for index, image in enumerate(unique_images):
        image_index_dict[image] = "img{}".format(index+1)

    for index, row in preannotations.iterrows():
        result = {
            "value": {
                "x": row['x'] / original_width * 100,
                "y": row['y'] / original_height * 100,
                "width": 0.75,
                "label": row["label"]
            },
            "to_name": image_index_dict[row["image"]],
            "type": "KeyPointLabels",
            "from_name": "keypoint_{}".format(image_index_dict[row["image"]]),
            "original_width": original_width,   
            "original_height": original_height,
        }
        predictions.append(result)
            
    return {"result": predictions}

def import_image_tasks(label_studio_project, local_image_dir, predictions=None):
    """
    Imports image tasks into a Label Studio project.

    Args:
        label_studio_project (LabelStudioProject): The Label Studio project to import tasks into.
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

def import_keypoint_tasks(label_studio_project, local_image_dir, predictions=None):
    """
    Imports image tasks into a Label Studio project.

    Args:
        label_studio_project (LabelStudioProject): The Label Studio project to import tasks into.
        local_image_dir (str): The local directory where the images are stored.
        predictions (dict, optional): A dictionary containing the predictions for the images. Key is image name e.g. 'DSC_2577.JPG'

    Returns:
        None
    """
    unique_images = predictions.image.unique()
    data_dict = {}
    for i, image_name in enumerate(unique_images):
        data_dict["img{}".format(i+1)] = os.path.join("/data/local-files/?d=input/", os.path.basename(image_name))    
    
    image_json = label_studio_keypoint_format(local_image_dir, predictions)

    upload_dict = {"data": data_dict, "predictions": [image_json]}
    label_studio_project.import_tasks(upload_dict)

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
        keypoints (dataframe): The keypoints to be drawn on images
    Returns:
        None
    """
    #Read each csv file and create a dictionary with the image name as the key and the dataframe as the value
    preannotations = {os.path.splitext(os.path.basename(preannotation))[0]: pd.read_csv(preannotation) for preannotation in preannotations}
    sftp_client = create_client(user=user, host=host, key_filename=key_filename)
    label_config = create_label_config(predictions=preannotations)
    project_name = os.path.basename(os.path.dirname(images[0]))
    label_studio_project = connect_to_label_studio(url=label_studio_url, project_name=project_name, label_config=label_config)
    labeled_images = draw_polygon_mask_on_images(images, keypoints)
    upload_images(sftp_client=sftp_client, images=labeled_images, folder_name=folder_name)
    import_image_tasks(label_studio_project=label_studio_project, local_image_dir=os.path.dirname(labeled_images[0]), predictions=preannotations)

def draw_keypoints_on_images(images, keypoints):
    """
    Draws keypoints on images using OpenCV.

    Args:
        images (list): List of image file paths.
        keypoints (dict): Dictionary where keys are image file names and values are lists of keypoints.
                            Each keypoint is a tuple (x, y).

    Returns:
        None
    """
    # Save in a new directory
    keypoint_image_dir = os.path.join(os.path.dirname(images[0]), "keypoints")
    os.makedirs(keypoint_image_dir, exist_ok=True)

    labeled_images = []
    for image_path in images:
        image = cv2.imread(image_path)
        image_name = os.path.basename(image_path)
        image_keypoints = keypoints[keypoints['image'] == os.path.basename(image_path)]
        for _, row in image_keypoints.iterrows():
            x, y, color = row['x'], row['y'], row['color']
            cv2.circle(image, center=(int(x), int(y)), radius=50, color=(int(color[0]), int(color[1]), int(color[2])), thickness=10)

        output_path = os.path.join(keypoint_image_dir, image_name)
        cv2.imwrite(output_path, image)
        labeled_images.append(output_path)

    return labeled_images

def draw_polygon_mask_on_images(images, keypoints):
    """
    Draws polygon masks on an image using OpenCV.

    Args:
        image_path (str): Path to the image file.
        polygons (list): List of polygons where each polygon is a list of points.

    Returns:
        str: Path to the output image with polygon masks.
    """
    # Save in a new directory
    
    keypoint_image_dir = os.path.join(os.path.dirname(images[0]), "keypoints")
    os.makedirs(keypoint_image_dir, exist_ok=True)


    # create a key of image and match_image, sorted
    keypoints['image'] = keypoints['image'].apply(lambda x: os.path.basename(x))
    keypoints['match_image'] = keypoints['match_image'].apply(lambda x: os.path.basename(x))
    keypoints['key'] = keypoints.apply(lambda x: '-'.join(sorted([x['image'], x['match_image']])), axis=1)  

    # Make a unique color per key
    def generate_bright_color():
        """Generate a bright color."""
        return tuple(random.randint(128, 255) for _ in range(3))

    colors = [generate_bright_color() for _ in range(len(keypoints['key'].unique()))]

    color_dict = dict(zip(keypoints['key'].unique(), colors))
    keypoints['color'] = keypoints['key'].apply(lambda x: color_dict[x])

    output_paths = []
    for image_path in images:
        image = cv2.imread(image_path)
        for match_image in images:
            if image_path == match_image:
                continue
        
            image_keypoints = keypoints[(keypoints['image'] == os.path.basename(image_path)) & (keypoints['match_image'] == os.path.basename(match_image))]
            points = image_keypoints[['x', 'y']].values.astype(np.int32)

            if len(points) > 0:
                hull = cv2.convexHull(points)
                color = image_keypoints['color'].values[0]
                cv2.polylines(image, [hull], isClosed=True, color=color, thickness=40)

        output_path = os.path.join(keypoint_image_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        output_paths.append(output_path)

    return output_paths

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
