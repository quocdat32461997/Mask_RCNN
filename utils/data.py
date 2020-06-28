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
        super(COCO_loader, self).__init__()
        self.data = COCO(annotation_path)
        self.image_path = image_path
        self.classes = self.get_classes(print_classes = True) # by default, get categories
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.data_augmentation = data_augmentation
        self.max_objects = max_objects

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


            return image, bboxes, masks
    def preproces(self, image_path, bboxes, masks):
        """
        preprocess - method to resize image and augment bbox and mask
        Inputs:
            image_path - path to image
            bboxes - numpy array of bounding boxes in shape of [instnaces, x1, y1, x2, y2, class]
            masks - numpy array of binary masks in shape of [instnaces, width, height]
        """
        image = cv2.imread(image_path)
        ih, iw = image.shape[:2]
        h, w = self.image_shape

        # shuffle bounding boxes and masks
        random_samples = np.random.shuffle(np.arange(len(bboxes.shape[0])))
        bboxes = bboxes[random_samples]
        masks = masks[random_samples]

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
                image = new_image
                image /= 255.0
                del new_image #remove unused variable

                #resize masks
                masks = cv2.resize(masks, (nw, nh), cv2.INTER_CUBIC)
                new_masks = np.zeros((masks.shape[0], h, w))
                new_masks[dy : dy + nh, dx + dx + nw, :] = masks
                masks = new_masks
                del new_masks

            # correct boxes
            if len(bboxes) > 0:
                if len(bboxes) > self.max_objects:
                    bboxes = bboxes[:self.max_objects, :]
                    masks = masks[:self.max_objects, :, :]

                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + dx
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + dy
                if len(bboxes) < self.max_objects:
                    pad_size = max(0, self.max_objects - len(bboxes))
                    np.concatenate((bboxes, np.zeros((pad_size, 5))))
                    np.concatenate((masks, np.zeros((pad_size, h, w))))
            return image, bboxes, masks

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
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
        new_image[dy : dy + nh, dx : dx + nw, :] = image
        image = new_image
        del new_image

        # resize masks
        masks = cv2.resize(masks, (nw, nh), cv2.INTER_CUBIC)
        new_masks = np.zeros((masks.shape[0], h, w))
        new_masks[dy : dy + nh, dx + dx + nw, :] = masks
        masks = new_masks
        del new_masks

        flip = rand() < 0.5
        if flip:
            # flip around x-axis
            new_image = cv2.flip(image, axis = 1)
            masks = cv2.flip(masks, axis = 1)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image = hsv_to_rgb(x) # numpy array, 0 to 1

        # make gray
        gray = rand() < .2
        if gray:
            image_gray = np.dot(image, [0.299, 0.587, 0.114])
            # a gray RGB image is GGG
            image = np.moveaxis(np.stack([image_gray, image_gray, image_gray]),0,-1)

        # invert colors
        invert = rand()< .1
        if invert:
            image_data = 1. - image_data

        # correct boxes
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * nw/w + dx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * nh/h + dy
            if flip: bboxes[:, [0, 2]] = w - bboxes[:, [0, 2]]
            bboxes[:, 0:2][bboxes[:, 0:2]<0] = 0
            bboxes[:, 2][bboxes[:, 2]>w] = w
            bboxes[:, 3][bboxes[:, 3]>h] = h
            bboxes_w = bboxes[:, 2] - bboxes[:, 0]
            bboxes_h = bboxes[:, 3] - bboxes[:, 1]
            bboxes = bboxes[np.logical_and(bboxes_w>1, bboxes_h>1)] # discard invalid box

            if len(bboxes) > self.max_objects:
                bboxes = bboxes[:self.max_objects, :]
                masks = masks[:self.max_objects, :, :]

            if len(bboxes) < self.max_objects:
                pad_size = max(0, self.max_objects - len(bboxes))
                np.concatenate((bboxes, np.zeros((pad_size, 5))))
                np.concatenate((masks, np.zeros((pad_size, h, w))))
                
        return image, bboxes, masks

    def rand(a=0, b=1):
        return np.random.rand()*(b-a) + a

    def annotation_parser(self, annotations):
        """
        annotation_parser - method to genrate binary mask and bboxes of annotations of an image
        Inputs:
            annotations - list of array of annotations
        Outputs:
            masks - numpy array of masks in shape [instances, width, height]
            bboxes - numpy array of masks in shape [isntnaces, x1, y1, x2, y2, class]
        """
        bboxes = []
        masks = []
        classes = []
        for ann in annotation:
            classes.apend(ann['category_id'])
            bboxes.append(ann['bbox'])
            masks.append(self.data.annToMask(ann))
        return np.array(mask), np.concatenate((np.array(bboxes), np.array(classes)), axis = 0)
