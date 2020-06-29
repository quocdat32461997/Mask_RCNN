import config as cfg
import os
from utils.data import COCOLoader

coco_train = COCOLoader(image_path = os.path.join(cfg.COCO_PATH, 'val2017'), annotation_path = os.path.join(cfg.COCO_PATH, 'annotations/instances_val2017.json'), data_augmentation = True)

for x in coco_train():
    print(type(x))
    input()
