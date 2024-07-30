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
from PIL import Image


'''
CALL
python --run_name name --pred_path path/to/pred --pred_path path/to/out --gt_path path/to/gt --shape shape
'''


parser = argparse.ArgumentParser()
parser.add_argument('--run_name', help='Path to the run folder', type=str)
parser.add_argument('--pred_path', help='Path to the pred folder', type=str)
parser.add_argument('--out_path', help='Path to the out folder', type=str)
parser.add_argument('--gt_path', help='Path to the gt folder', type=str)
parser.add_argument('--shape', help='img_shape', type=int)
parsed_args = parser.parse_args()

run_name = parsed_args.run_name
out_path = parsed_args.out_path
pred_path = parsed_args.pred_path
gt_path = parsed_args.gt_path
shape = parsed_args.shape

IMG_WIDTH = shape
IMG_HEIGHT = shape

grey_list = sorted(os.listdir(pred_path))
grey = np.zeros((len(grey_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(grey_list):
    path = os.path.join(pred_path, id_)
    # img = imread(path, as_gray = True)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    grey[n] = img

gt_list = sorted(os.listdir(gt_path))
gt = np.zeros((len(gt_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(gt_list):
    path = os.path.join(gt_path, id_)
    # img = imread(path, as_gray = True)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    gt[n] = img

grey_flat = grey.flatten()
gt_flat = gt.flatten()
gt_flat = np.where(gt_flat>100, 1, 0)

zeros = np.count_nonzero(gt_flat == 0)
ones = np.count_nonzero(gt_flat == 1)

roc_auc = metrics.roc_auc_score(gt_flat, grey_flat) #  shape (n_samples,)

recall_list = list()
precision_list = list()
fallout_list = list()
accuracy_list =  list()
f1_list = list()

max_grey = np.max(grey_flat)

for thr in tqdm(range(1, max_grey)):  # range(1, max_grey)
    bw_flat = np.where(grey_flat>thr, 1, 0)

    try:
        n_classes = 2 # Binaric segmentation
        y = n_classes * gt_flat + bw_flat
        y = np.bincount(y, minlength=n_classes*n_classes)
        matrix_confusion = y.reshape(n_classes, n_classes)
        TN, FP, FN, TP = matrix_confusion.ravel()
        is_finished = True
    except ValueError as e:
        is_finished = False
        print(e)
        print(f'GTS = {np.unique(gt_flat)}, pred = {np.unique(bw_flat)}')
        print(f'Y = {y}')

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    fallout = FP / (FP + TN)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    f1 = 2 * ((precision * recall) / (precision + recall))

    recall_list.append(recall)
    precision_list.append(precision)
    fallout_list.append(fallout)
    accuracy_list.append(accuracy)
    f1_list.append(f1)


thr_best = np.nanargmax(f1_list)

acc_best = accuracy_list[thr_best]
prec_best = precision_list[thr_best]
rec_best = recall_list[thr_best]
fallout_best = fallout_list[thr_best]
f1_best = f1_list[thr_best]

save_path = os.path.join(out_path, "metrics")

try:
    os.mkdir(save_path)
except:
    print("")


data = {'Run': [run_name], 'thr': [thr_best], 'acc': [acc_best], 'prec': [prec_best], 'rec': [rec_best], 'fall': [fallout_best], 'f1': [f1_best], 'auc': [roc_auc]}

df = pd.DataFrame(data)
print(df)

df.to_excel(os.path.join(save_path,'metrics.xlsx'))



 

