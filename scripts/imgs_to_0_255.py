import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, imshow, imsave


"""

python imgs_to_0_255.py --in_path /mnt/c/Users/haddo/Halimeda/merged_model_0/val/gt_semantic \
                    --shape 1024 \
                    --sp /mnt/c/Users/haddo/Halimeda/merged_model_0/val/gt_ss

"""

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', help='Path to the run folder', type=str)
parser.add_argument('--sp', help='save_path', type=str)
# parser.add_argument('--shape', help='img_shape', type=int)
parsed_args = parser.parse_args()

in_path = parsed_args.in_path
sp = parsed_args.sp
# shape = parsed_args.shape

# IMG_WIDTH = shape
# IMG_HEIGHT = shape

grey_list = sorted(os.listdir(in_path))

print("grey_list", grey_list)

# grey = np.zeros((len(grey_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(grey_list):
    id, extension = id_.split('.')
    path = os.path.join(in_path, id_)
    img = imread(path)[:,:,:3]
    print(img.shape)
    img_new = np.zeros(img.shape, dtype=np.uint8)
    print(img_new.shape)
    imsave(os.path.join(sp, id+'_gt.jpg'), img_new)
