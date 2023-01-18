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
"CALL: python3 evaluation.py --pred_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/hyp_high_lr2_a/inference_test/coverage_pred \
      --gt_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/hyp_high_lr2_a/inference_test/coverage_gt --shape 1024 \
        --run_name yolo_XL_hyp_high_lr2_a --save_path /mnt/c/Users/haddo/yolov5/projects/halimeda/final_trainings/yolo_XL/hyp_high_lr2_a/inference_test "

 """
parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', help='Path to the run folder', type=str)
parser.add_argument('--gt_path', help='Path to the mask folder', type=str)
parser.add_argument('--run_name', help='Path to the mask folder', type=str)
parser.add_argument('--save_path', help='Path to the mask folder', type=str)
parser.add_argument('--shape', help='img_shape', type=int)


parsed_args = parser.parse_args()

pred_path = parsed_args.pred_path
gt_path = parsed_args.gt_path
run_name = parsed_args.run_name
save_path = parsed_args.save_path
shape = parsed_args.shape

IMG_WIDTH = shape
IMG_HEIGHT = shape

grey_list = sorted(os.listdir(pred_path))
img = imread(os.path.join(pred_path, grey_list[0]))

grey = np.zeros((len(grey_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(grey_list):
    path = os.path.join(pred_path, id_)
    img = imread(path, as_gray = True)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    grey[n] = img

mask_list = sorted(os.listdir(gt_path))
mask = np.zeros((len(mask_list), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
for n, id_ in enumerate(mask_list):
    path = os.path.join(gt_path, id_)
    img = imread(path,as_gray = True)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask[n] = img

grey_flat = grey.flatten()
mask_flat = mask.flatten()
mask_flat = np.where(mask_flat>100, 1, 0)

zeros = np.count_nonzero(mask_flat == 0)
ones = np.count_nonzero(mask_flat == 1)

fp, tp, thr = metrics.roc_curve(mask_flat,grey_flat)
roc_auc = metrics.roc_auc_score(mask_flat, grey_flat) #  shape (n_samples,)

#plt.plot(fp,tp)
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()

recall_list = list()
precision_list = list()
fallout_list = list()
accuracy_list =  list()
f1_list = list()

#thr_list = [100,150]
max_grey = np.max(grey_flat)

for thr in tqdm(range(1, max_grey)):  # range(1, max_grey)

    bw_flat = np.where(grey_flat>thr, 1, 0)

    TN, FP, FN, TP = metrics.confusion_matrix(mask_flat,bw_flat).ravel()

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    fallout = FP/(FP+TN)
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    f1 = 2*((precision*recall)/(precision+recall))

    recall_list.append(recall)
    precision_list.append(precision)
    fallout_list.append(fallout)
    accuracy_list.append(accuracy)
    f1_list.append(f1)


thr_best = np.argmax(f1_list)

acc_best = accuracy_list[thr_best]
prec_best = precision_list[thr_best]
rec_best = recall_list[thr_best]
fallout_best = fallout_list[thr_best]
f1_best = f1_list[thr_best]


try:
    os.mkdir(save_path)
except:
    print("")


data = {'Run': [run_name], 'thr': [thr_best], 'acc': [acc_best], 'prec': [prec_best], 'rec': [rec_best], 'fall': [fallout_best], 'f1': [f1_best], 'auc': [roc_auc]}

df = pd.DataFrame(data)
print(df)

df.to_excel(os.path.join(save_path,'metrics.xlsx'))



 

