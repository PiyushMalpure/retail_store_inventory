# retail_store_inventory
An object detection and recognition project for detecting  inventory in a retail store

## Setup Environment
1. clone the repository
2. git submodule init
3. git submodule update
4. cd retail_store_inventory
5. In a python or conda environment install the requirements.txt

## TASK 1: Object Detection

Ensure that the IMAGE to query is inside the data folder and path is correctly given in the bounding_box_detection.py line 44.
1. cd src
2. run python bounding_box_detection.py

This step creates the bounding boxes around the given query image for all detected objects. The result image is stored in the results folder.
It also creates the cropped images for each bounding box for TASK 2.

## TASK 2: Product Identification

Ensure the reference images are inside data/images_inventory.
1. cd utils
2. python create_pairs_txt.py
This step will create the input required for the product identification.
Ensure paths in the file are correct. 

1. cd src 
2. run match_pairs.py

This will create a txt file in results folder, consisting of each cropped image mapped with the best matching product from the product images.

## OPTIONAL: 
### Run repository using Docker 

1. docker build -t retail_store_inventory
2. docker run -it retail_store_inventory src/bounding_box_detection.py

Follow the above steps to run the tasks in docker.

TODO:
Create a final run file. Due to time constraints could not add up everything under one run file and test it.