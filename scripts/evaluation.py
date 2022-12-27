import os
import cv2
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
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

grey_flat = grey.flatten()
mask_flat = mask.flatten()
zeros = np.count_nonzero(mask_flat == 0)
ones = np.count_nonzero(mask_flat == 1)

fp, tp, thr = metrics.roc_curve(mask_flat,grey_flat)
fn = ones-tp
tn = zeros - fp

#plt.plot(fp,tp)
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()

acc = (tp+tn)/(zeros+ones)
prec = (tp)/(tp+fp)
rec = (tp)/(ones)
fallout = (fp)/(zeros)
f1 = 2*((prec*rec)/(prec+rec))


thr_best = np.argmax(f1)
acc_best = acc[thr_best]
prec_best = prec[thr_best]
rec_best = rec[thr_best]
fallout_best = fallout[thr_best]
f1_best = f1[thr_best]
roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score) #  shape (n_samples,)




 

