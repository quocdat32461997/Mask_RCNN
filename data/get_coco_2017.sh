#!/bin/bash
# Scrip to download COCO 2017 for Instance Segmentation

# Create folder coco
mdkri coco
wget -c http://images.cocodataset.org/zips/train2017.zip -P coco
wget -c http://images.cocodataset.org/zips/val2017.zip -P coco
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P coco

# Unzip files
unzip coco/train2017.zip -d ./coco
unzip coco/val2017.zip -d ./coco
unzip coco/annotations_trainval2017.zip -d ./coco

# Parse data
#python3 parse_coco.py --file_path coco/annotations/instances_train2017.json --image_path coco/train2017 --output_path coco/train.json
#python3 parse_coco.py --file_path coco/annotations/instances_val2017.json --image_path coco/val2017 --output_path coco/val.json

# download and install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
sed -i 's/python/python3/g' Makefile
sudo make install
