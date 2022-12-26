import os
import cv2
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imshow, imsave

run = "1024_8_default"
save_path = os.path.join("/home/tintin/SS_Halimeda/runs", run)
path_grey = os.path.join(save_path,run,"inference_out/grey")
path_mask = "/home/tintin/SS_Halimeda/data/splits/base/test/mask"

grey_list = sorted(os.listdir(path_grey))
img = imread(os.path.join(path_grey, grey_list[0]))
IMG_HEIGHT, IMG_WIDTH = np.shape(img)

grey = np.zeros((len(grey_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(grey_list):
    path = os.path.join(path_grey, id_)
    img = imread(path)
    grey[n] = img

mask_list = sorted(os.listdir(path_mask))
mask = np.zeros((len(mask_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(mask_list):
    path = os.path.join(path_mask, id_)
    img = imread(path)
    mask[n] = img

acc_list = list()
prec_list = list()
rec_list = list()
fallout_list = list()
f1_list = list()
matrix_list = list()

for thr in range(256):

    grey_t = (grey > thr)

    for idx in range(len(grey_list)):
        # compare grey_t[idx] con maks[idx]
        # ir generando conf matrix
    # append matrix
    # calcular acc prec rec fall f1
    # append acc prec rec fall f1

# select best metrics -> best thr

# flaten grey and mask
# roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score)


 

