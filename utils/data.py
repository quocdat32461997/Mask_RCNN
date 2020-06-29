"""
data.py - module to load data
"""

""" import dependencies """
import tensorflow as tf
import json
import cv2
import os
import numpy as np
from pycocotools.coco import COCO
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from utils.utils import rand

class COCOLoader:
    """
    COCO - data loader class to load data
    """
    def __init__(self, image_path, annotation_path, max_objects = 20, batch_size = 16, image_shape = (416, 416),
        jitter = .3, hue = .1, sat = 1.5, val = 1.5, data_augmentation = True, proc_image = True):
        """
        __init__ - funciton to initialize coco class
        Inputs:
            image_path  - path to image foler
            annotation_path - path to annotation file
            batch_size - size of batches
            image_shape - shape of images
        """
        self.data = COCO(annotation_path)
        self.image_path = image_path
        self.classes = self.get_classes(print_classes = True) # by default, get categories
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.data_augmentation = data_augmentation
        self.max_objects = max_objects
        self.proc_image = proc_image

        # parameters for augmenting images
        self.jitter = jitter
        self.hue = hue
        self.sat = sat
        self.val = val

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
            image_path = os.path.join(self.image_path, info['file_name'])

            # get annotations
            annIds = self.data.getAnnIds(imgIds = [id])
            masks, bboxes = self.annotation_parser(self.data.loadAnns(annIds))

            # preprocess image
            image, bboxes, masks = self.preprocess(image_path, bboxes, masks)

            #print("Shape of Image {}, Bounding Boxes {}, and Masks {}".format(image.shape, bboxes.shape, masks.shape))
            #print("Type of Image {}, Bounding Boxes {}, and Masks {}".format(image.dtype, bboxes.dtype, masks.dtype))
            #input()
            yield image, bboxes, masks

    def __call__(self):
        return tf.data.Dataset.from_generator(self.generator,
            output_types = (tf.float64, tf.float64, tf.float64),
            output_shapes = (tf.TensorShape([self.image_shape[0], self.image_shape[1], 3]), tf.TensorShape([self.max_objects, 5]), tf.TensorShape([self.max_objects, self.image_shape[0], self.image_shape[1]])))

    def preprocess(self, image_path, bboxes, masks):
        """
        preprocess - method to resize image and augment bbox and mask
        Inputs:
            image_path - path to image
            bboxes - numpy array of bounding boxes in shape of [instnaces, x1, y1, x2, y2, class]
            masks - list of binary masks in shape of [instnaces, width, height]
        """
        image = cv2.imread(image_path)
        ih, iw = image.shape[:2]
        h, w = self.image_shape

        # shuffle bounding boxes and masks
        #random_samples = np.random.shuffle(np.arange(bboxes.shape[0]))
        #bboxes = bboxes[random_samples]
        #masks = masks[random_samples]

        if not self.data_augmentation:
            # resize image
            scale = min(w / iw, h / ih)
            nh, nw = int(ih * scale), int(iw * scale)
            dy, dx = (h - nh) // 2, (w - nw) // 2

            if self.proc_image:
                #resize image
                image = cv2.resize(image, (nw, nh), cv2.INTER_CUBIC)
                new_image = np.ones((h, w, 3)) * 128 #Image.new('RGB', (w, h), (128, 128, 128))
                new_image[dy : dy + nh, dx : dx + nw, :] = image
                new_image /= 255.0
                del image #remove unused variable

                #resize masks
                masks = cv2.resize(masks, (nw, nh), cv2.INTER_CUBIC)
                new_masks = np.zeros((h, w, masks.shape[-1]))
                new_masks[dy : dy + nh, dx : dx + nw, :] = masks
                new_masks = new_masks.reshape((-1, new_masks.shape[0], new_masks.shape[1])) #reshaep to [instances, height, width]
                del masks

            # correct boxes
            if len(bboxes) > 0:
                if len(bboxes) > self.max_objects:
                    bboxes = bboxes[:self.max_objects, :]
                    masks = masks[:self.max_objects, :, :]

                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + dx
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + dy
                if len(bboxes) < self.max_objects:
                    pad_size = max(0, self.max_objects - len(bboxes))
                    bboxes = np.concatenate((bboxes, np.zeros((pad_size, 5))))
                    new_masks = np.concatenate((new_masks, np.zeros((pad_size, h, w))))
            return new_image, bboxes, new_masks

        # resize image
        new_ar = w/h * rand(1-self.jitter,1+self.jitter)/rand(1-self.jitter,1+self.jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))

        # resize image
        image = cv2.resize(image, (nw,nh), cv2.INTER_CUBIC)
        new_image = np.ones((h, w, 3)) * 128
        new_image[max(0, dy) : min(h, dy + nh), max(0, dx) : min(w, dx + nw), :] = image[max(0, -dy) : min(nh, h - dy), max(0, -dx) : min(nw, w - dx), :]
        del image

        # resize masks
        masks = cv2.resize(masks, (nw, nh), cv2.INTER_CUBIC)
        new_masks = np.zeros((h, w, masks.shape[-1]))
        new_masks[max(0, dy) : min(h, dy + nh), max(0, dx) : min(w, dx + nw), :] = masks[max(0, -dy) : min(nh, h - dy), max(0, -dx) : min(nw, w - dx), :]
        del masks

        flip = rand() < 0.5
        if flip:
            # flip around x-axis
            new_image = cv2.flip(new_image, 1)
            new_masks = cv2.flip(new_masks, 1)

        # distort image
        hue = rand(-self.hue, self.hue)
        sat = rand(1, self.sat) if rand()<.5 else 1/rand(1, self.sat)
        val = rand(1, self.val) if rand()<.5 else 1/rand(1, self.val)
        x = rgb_to_hsv(np.array(new_image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        new_image = hsv_to_rgb(x) # numpy array, 0 to 1

        # make gray
        gray = rand() < .2
        if gray:
            image_gray = np.dot(new_image, [0.299, 0.587, 0.114])
            # a gray RGB image is GGG
            new_image = np.moveaxis(np.stack([image_gray, image_gray, image_gray]),0,-1)

        # invert colors
        invert = rand()< .1
        if invert:
            new_image = 1. - new_image

        # correct boxes
        new_masks = new_masks.reshape((-1, new_masks.shape[0], new_masks.shape[1])) #reshaep to [instances, height, width]
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * nw/iw + dx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * nh/ih + dy
            if flip: bboxes[:, [0, 2]] = w - bboxes[:, [0, 2]]
            bboxes[:, 0:2][bboxes[:, 0:2]<0] = 0
            bboxes[:, 2][bboxes[:, 2]>w] = w
            bboxes[:, 3][bboxes[:, 3]>h] = h
            bboxes_w = bboxes[:, 2] - bboxes[:, 0]
            bboxes_h = bboxes[:, 3] - bboxes[:, 1]

            valid_instances = np.logical_and(bboxes_w > 1, bboxes_h > 1)
            bboxes = bboxes[valid_instances] # discard invalid box
            new_masks = new_masks[valid_instances]

            #print("Bboxs shape {}".format(bboxes.shape))
            #print("Mask shape {}".format(new_masks.shape))
            if len(bboxes) > self.max_objects:
                bboxes = bboxes[:self.max_objects, :]
                new_masks = new_masks[:self.max_objects, :, :]
            #print("Mask shape {}".format(new_masks.shape))
            #print("Bboxs shape {}".format(bboxes.shape))
            if len(bboxes) < self.max_objects:
                pad_size = max(0, self.max_objects - len(bboxes))
                bboxes = np.concatenate((bboxes, np.zeros((pad_size, 5))))
                new_masks = np.concatenate((new_masks, np.zeros((pad_size, h, w))))
            #print("Mask shape {}".format(new_masks.shape))
            #print("Bboxs shape {}".format(bboxes.shape))
            #input()
        return new_image, bboxes, new_masks

    def annotation_parser(self, annotations):
        """
        annotation_parser - method to genrate binary mask and bboxes of annotations of an image
        Inputs:
            annotations - list of array of annotations
        Outputs:
            masks - lists masks in shape [instances, width, height]
            bboxes - numpy array of masks in shape [isntnaces, x1, y1, x2, y2, class]
        """
        bboxes = []
        masks = []
        classes = []
        for ann in annotations:
            classes.append(ann['category_id'])
            bboxes.append(ann['bbox'])
            masks.append(self.data.annToMask(ann))
        masks = np.array(masks)
        masks = masks.reshape((masks.shape[1], masks.shape[2], -1))
        classes = np.expand_dims(np.array(classes), axis = 1)
        bboxes = np.concatenate((np.array(bboxes), classes), axis = 1)
        return masks, bboxes
