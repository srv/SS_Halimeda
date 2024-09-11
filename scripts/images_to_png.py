import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--data_in_path', help='Path to the data folder', type=str)
parser.add_argument('--data_out_path', help='Path to the data folder', type=str)
parsed_args = parser.parse_args()

data_in_path = parsed_args.data_in_path
data_out_path = parsed_args.data_out_path

images_PATH = os.path.join(data_in_path, "images")
masks_PATH =  os.path.join(data_in_path, "gt")
out_images_PATH = os.path.join(data_out_path, "images")
out_masks_PATH =  os.path.join(data_out_path, "gt")

images_list = sorted(os.listdir(images_PATH))
masks_list = sorted(os.listdir(masks_PATH))

# train images and masks
for n, id_ in enumerate(images_list):
    path = os.path.join(images_PATH, id_)
    im = Image.open(path)
    new_name = id_.split('.')[0] + '.png'
    path = os.path.join(out_images_PATH, new_name)
    im.save(path)

for n, id_ in enumerate(masks_list):
    path = os.path.join(masks_PATH, id_)
    im = Image.open(path)
    new_name = id_.split('.')[0] + '.png'
    path = os.path.join(out_masks_PATH, new_name)
    im.save(path)
