# Double Counting for Airborne Ecological Object Detection

A repo for visualizing airborne imagery, making a prediction with a object detection model and remove double counting among overlapping images.

# Motivation

Wildlife surveys often collect sporadic image data of large numbers of organisms congregated in small areas such as bird colonies, coral reefs, and grazing herds. The task is to detect and classify the number of unique individuals in a set of images, which lack georeferencing metadata or other auxiliary sources of location information. This work differs from the common drone collect and orthomosaic image collection in which highly overlapping, often georeferenced imagery is collected for the purpose of creating a single mosaic. There appears to be a large number of potential applications for reducing double counting in image sets both within and outside of ecology, although many papers seem to skirt the issue by create artificial gaps in image series (e.g. [here](https://www.mdpi.com/2072-4292/13/16/3276)). The challenges lie in images being collected with uneven overlap, potential inclusion of spurious images within collections that should be discarded, and the need to account for small perturbations in object appearance. Moreover, landscapes are convoluted with fine-grained detail differences. Assumptions include the rapid sequence in which images are collected and that images follow a reasonably smooth trajectory without jumping from side to side. Strategies involve techniques like "Stitch and predict," where overlapping image pairs are stitched together using keypoints to create a mosaic, onto which an object detection pipeline is applied. Another strategy, "Predict and delete", first predicts detections in each raw image, and then iteratively chooses boxes based on stitch membership. Finally we could "Predict and verify," by predicting on raw data, generating local features for each prediction, and searching all other predictions to remove boxes based on a similarity matrix. Key literature references include DeepFeatures, DeepTracking, and Background features with attention, which employ deep learning techniques for image analysis and object tracking, contributing to automated ecological monitoring. However, these methods require careful development and may involve significant time investment.

# Installation

```
conda create python=3.10 --name DoubleCounting
pip install -r requirements.txt
```
# What does this repo do?

* Reconstruct a structure-from-motion model of overlapping images

```
# Create a SfM model using hloc and pycolmap
output_path = Path(os.path.join(test_data_dir, "output_birds"))
output_path.mkdir(parents=True, exist_ok=True)
image_dir = Path(os.path.join(os.path.dirname(__file__),"data/birds"))
references = [str(p.relative_to(image_dir)) for p in (image_dir).iterdir()]
create_sfm_model(image_dir=image_dir, references=references, output_path=output_path)
```

* Take a [DeepForest](https://deepforest.readthedocs.io/en/latest/) model and predict on each image. This can be easily swapped for any common objection tool and conform to the output data structure.

```
from scripts.predict import predict
_, predictions = predict(image_paths=image_paths, save_dir=None, model_path=None)
```

* Iteratively loop through image predictions and remove double counted detections based on image homography

```
from scripts.stitching import align and delete
final_predictions = align_and_delete(predictions=predictions, model=model_birds, image_dir=image_dir, matching_h5_file=output_path / "matches.h5", strategy=strategy)
```

* Visualize

```
# Show original (Blue) and filtered (Pink) predictions
fig, axs = pyplot.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()
images_to_plot = final_predictions["image_path"].unique()
images_to_plot.sort()
for i, image_path in enumerate(images_to_plot):
    image = cv2.imread(os.path.join(image_dir.__str__(), image_path))
    # Show originals as well
    original_predictions = predictions[predictions["image_path"] == image_path]
    for index, row in original_predictions.iterrows():
        x = row["xmin"]
        y = row["ymin"]
        w = row["xmax"] - row["xmin"]
        h = row["ymax"] - row["ymin"]
        # Draw bounding box
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 7)
    unique_predictions_image = final_predictions[final_predictions["image_path"] == image_path]
    for index, row in unique_predictions_image.iterrows():
        x = row["xmin"]
        y = row["ymin"]
        w = row["xmax"] - row["xmin"]
        h = row["ymax"] - row["ymin"]
        # Draw bounding box
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (182, 192, 255), 5)
    axs[i].imshow(image[:,:,::-1])
    axs[i].set_title(f"Final predictions for {image_path}")

pyplot.tight_layout()
pyplot.show()
```

Within this workflow there are steps to compute among image homography, matching features, warping points given a homographic matrix and other utility functions.

# Next steps

This repo has only been lightly tested for one use case. All thoughts are welcome.

- [ ] Diversify the experimental dataset (please submit an issue with data)
- [ ] Generate evaluation metrics 

# Testing

```
pytest tests/
```

For example to test the 2d matching pipeline

```
pytest tests/test_stitching.py -k test_align_and_delete
```

## Examples

## 2d homography
In this example there are four images of a bird colony. In blue are the original object detections for class 'Bird', in pink are the retained detections after double counting. Double click to zoom in.

![](public/example.png)

## 3d reconstruction
Here we label in green two points in the raw images and find them in the 3d space
![](public/example_result.png)

### Relevant Literature

Double Counting in Livestock imagery

Shao, W., R. Kawakami, R. Yoshihashi, S. You, H. Kawase, and T. Naemura. 2020. Cattle detection and counting in UAV images based on convolutional neural networks. International Journal of Remote Sensing 
41:31–52.

Soares, V. H. A., M. A. Ponti, R. A. Gonçalves, and R. J. G. B. Campello. 2021. Cattle counting in the wild with geolocated aerial images in large pasture areas. Computers and Electronics in Agriculture 189:106354.

Double Counting in Agricultural Images

Xia, X., X. Chai, N. Zhang, Z. Zhang, Q. Sun, and T. Sun. 2022. Culling Double Counting in Sequence Images for Fruit Yield Estimation. Agronomy 12:440.

DeepFeatures: https://openaccess.thecvf.com/content/CVPR2021W/IMW/html/Efe_DFM_A_Performance_Baseline_for_Deep_Feature_Matching_CVPRW_2021_paper.html

DeepTracking: https://ieeexplore.ieee.org/abstract/document/8766896?casa_token=d4ngee5cFQ4AAAAA:vmNb96Y4hAlTRCE37aYgpEieF5ySgAfWWYeQwx0K8E7j8NGa1dP4m1OJJnxsmT01XI-OtzzI

Background features with attention: https://openaccess.thecvf.com/content/ICCV2021W/DSC/papers/Bansal_Where_Did_I_See_It_Object_Instance_Re-Identification_With_Attention_ICCVW_2021_paper.pdf
