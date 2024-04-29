# DoubleCounting
A repo for visualizing airborne imagery, making a prediction with a object detection model and remove double counting among overlapping images.

<img src="tests/data/birds/DSC_2520.JPG" alt="Sample Image 1" width="50%">
<img src="tests/data/birds/DSC_2522.JPG" alt="Sample Image 2" width="50%">

# Installation

```
conda create python=3.10 --name DoubleCounting
pip install -r requirements.txt
```
# What does this repo do?

* Remove

# Testing

```
pytest tests/
```

For example to test the pycolmap reconstruction and alignment run

```
pytest tests/test_stitching.py -k test_transform_2d_to_3d_simple
```

Here we label in green two points in the raw images and find them in the 3d space
![](public/example_result.png)


