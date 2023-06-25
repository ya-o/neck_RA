# Bone Project

## Usage

#### Environment
Please first make the environment following the MMpose [instruction](https://mmpose.readthedocs.io/en/latest/installation.html).
After that install the following package:
```
pip install opencv-python
pip install pandas
pip install matplotlib
```

#### Data Preparation
All setting can be modified in file json_generation.py.  
A json file need to be pre-made for training from a csv file and image folder.  
csv file: refer to annotation.csv.  
Image folder: should with images in the direction "data/NECK/images/neck" + str(pic_id) + ".jpg"  
We also manually set the cross-validation (see "split" in line 52, we currently set 10 folds.)  
"index" in line 53 can set one fold for validation and others for training (default as 0, use first folder).  
After setting, generate the json file by running command:
```
python json_generation.py
```

#### Training
For training, we default using model as "hrnet_w32_coco_tiny_256x192".
The total epoch can be set in line 262 (default as 40).
And run the command:
```
python train_own.py
```
During training, we make an evaluation on validation after each epoch with 'PCK', 'NME', "AUC".


#### Inference
And run the command:
```
python inference.py
```
Showing two image of "Pose Estimation" and "Truth". Note that the color of point and connection can be modified in file 
"configs/base/datasets/custom.py"