import os
import numpy as np
from skimage.io import imread, imsave

run = "1/000033"
save_path = os.path.join("/home/olivia/Halimeda/semantic_segmentation/SS_Halimeda/runs")
path_in = os.path.join(save_path,run,"inference")
path_out = os.path.join(save_path,run,"inference_thr")

try:
    os.mkdir(path_out)
except:
    print("")

im_list = sorted(os.listdir(path_in))
thr = 128

for im in im_list:
    path = os.path.join(path_in, im)
    grey= imread(path)
    grey_thr = np.where(grey < thr, 0, 255)
    imsave(os.path.join(path_out, im), grey_thr)
