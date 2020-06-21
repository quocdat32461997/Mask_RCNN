import argparse
import json
import os

def main(args):
    file_path = args.file_path

    with open(file_path, 'r') as file:
        json_file = json.load(file)

    annotations = {}

    # Get info of image
    for img in json_file['images']:
        print("Processing {}".format(img['file_name']))
        data = {
            'file_name' : os.path.join(args.image_path, img['file_name']),
            'height' : img['height'],
            'width' : img['width'],
            'annotations' : []
        }
        annotations[img['id']] = data

    # Get annotations of image
    for img in json_file['annotations']:
        annotations[img['image_id']]['annotations'].append({
            'class' : img['category_id'],
            'bbox' : img['bbox'],
            'segmentation' : img['segmentation']
        })

    # write to json file
    with open(args.output_path, 'w') as file:
        json.dump(annotations, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Preprocess COCO 2017 data for Instance Segmentation.')

    parser.add_argument('--file_path', type = str, help = 'Path to json-like file for annotations')
    parser.add_argument('--image_path', type = str, help = 'Path to image folder')
    parser.add_argument('--output_path', type = str, help = 'Path to store annotation data')
    main(parser.parse_args())
