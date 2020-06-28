"""
data.py - module to load data
"""

""" import dependencies """
import tensorflow as tf
from pycocotools.coco import COCO
import json
import cv2

class COCO_loader:
    """
    COCO_loader - data loader class to load data
    """
    def __init__(self, image_path, annotation_path, batch_size = 16, image_shape = (416, 416)):
        """
        __init__ - funciton to initialize coco class
        Inputs:
            image_path  - path to image foler
            annotation_path - path to annotation file
            batch_size - size of batches
            image_shape - shape of images
        """
        super(COCO_loader, self).__init__()
        self.data = COCO(annotation_path)
        self.image_path = image_path
        self.classes = self.get_classes(print_classes = True) # by default, get categories
        self.batch_size = batch_size
        self.image_shape = image_shape

    def get_classes(self, print_classes = False, supercategory = False):
        """
        get_classes - method to get class/categories or supercategories
        Inputs:
            print_classes - flag to print classes or not
            supercategory - flag to get supercategories
        Outputs:
            classes - list of categoires/clases
        """
        cats = self.data.loadCats(self.data.getCatIds())
        if not supercategory:
            classes = [cat['name'] for cat in cats]
            if print_classes:
                print('COCO categories: \n{}\n'.format(' '.join(classes)))
        else:
            classes = [cat['supercategories'] for cat in cats]
            if print_classes:
                print('COCO supercategories: \n{}\n'.format(' '.join(classes)))
        return classes

    def generator(self):
        """
        generator - method to return image, bbox, class, and mask dynamically
        Inputs: None
        Outputs:
            image - numpy array of iamge
            bbox - numpy array of shape [instnaces, x1, y1, x2, y2, class]
            masks - numpy array of shape [instances, width, height]
        """

        for id, info in self.data.imgs.items():
            # read image
            image = cv2.imread(os.path.join(self.image_path, info['file_name']))

            # get annotations
            annIds = self.data.getAnnIds(imgIds = [id])
            masks, bboxs = self.annotation_parser(self.data.loadAnns(annIds))

            # preprocess image
    def preproces(self, image, bboxes, mask):
        
    def annotation_parser(self, annotations):
        """
        annotation_parser - method to genrate binary mask and bboxes of annotations of an image
        Inputs:
            annotations - list of array of annotations
        Outputs:
            masks - numpy array of masks in shape [instances, width, height]
            bboxes - numpy array of masks in shape [isntnaces, x1, y1, x2, y2, class]
        """
        bboxs = []
        masks = []
        classes = []
        for ann in annotation:
            classes.apend(ann['category_id'])
            bboxs.append(ann['bbox'])
            masks.append(self.data.annToMask(ann))
        return np.array(mask), np.concatenate((np.array(bboxs), np.array(classes)), axis = 0)
