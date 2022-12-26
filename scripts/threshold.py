import os
import cv2
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imshow, imsave

run = "1024_8_default"
save_path = os.path.join("/home/tintin/SS_Halimeda/runs", run)
path_in = os.path.join(save_path,run,"inference_out/grey")
path_out = os.path.join(save_path,run,"inference_out/thr")

try:
    os.mkdir(path_out)
except:
    print("")

im_list = sorted(os.listdir(path_in))
img = imread(os.path.join(path_in, im_list[0]))
IMG_HEIGHT, IMG_WIDTH = np.shape(img)

grey = np.zeros((len(im_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(im_list):
    path = os.path.join(path_in, id_)
    img = imread(path)
    grey[n] = img


    grey_t = (grey > 128)

    base, ext = os.path.splitext(id_)
    base = base.split("_")[:-1]
    name = ''.join(base)
    imsave(os.path.join(path_out, name + "_thr" + ext), np.squeeze(grey_t[n]))