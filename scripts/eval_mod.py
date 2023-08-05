import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from natsort import natsorted

'''
CALL:
python3 eval.py --run_name 000033 --path_pred ../runs/1/000033/inference --path_out ../runs/1/000033/ --path_gt ../data/test/mask

python3 eval.py --run_name eval_low9_da_test_mikieval --path_pred  /mnt/c/Users/haddo/yolov5/projects/halimeda/NEW_DATASET/low9_da/2/inference_test/coverage/ \
      --path_out /mnt/c/Users/haddo/yolov5/projects/halimeda/NEW_DATASET/low9_da --path_gt /mnt/c/Users/haddo/yolov5/datasets/halimeda/NEW_DATASET/labels/test_coverage/

'''
def conditional_div(a, b):
    return a / b if b else 0


parser = argparse.ArgumentParser()
parser.add_argument('--run_name', help='Path to the run folder', type=str)
parser.add_argument('--path_pred', help='Path to the pred folder', type=str)
parser.add_argument('--path_out', help='Path to the out folder', type=str)
parser.add_argument('--path_gt', help='Path to the gt folder', type=str)
parsed_args = parser.parse_args()

run_name = parsed_args.run_name
path_out = parsed_args.path_out
path_pred = parsed_args.path_pred
path_gt = parsed_args.path_gt

 

image_extensions = ['.jpg', '.jpeg', '.png']  

pred_files = [filename for filename in os.listdir(path_pred) if filename.lower().endswith(tuple(image_extensions))]
pred_list = natsorted(pred_files)

gt_files = [filename for filename in os.listdir(path_gt) if filename.lower().endswith(tuple(image_extensions))]
gt_list = natsorted(gt_files)

confMatrix=np.zeros((2,2))

precision_list = list()
recall_list = list()
fallout_list = list()
f1_list =  list()
accuracy_list =  list()

preds = list()
gts = list()

print("evaluating: " + path_out)

for i in range(len(pred_list)):

    pred_file = os.path.join(path_pred, pred_list[i])
    gt_file = os.path.join(path_gt, gt_list[i])

    pred = cv2.imread(pred_file,2)
    gt = cv2.imread(gt_file,2)

    preds.append(pred)
    gts.append(gt)

for thr in tqdm(range(255)):
	

    TP_TOTAL = 0
    TN_TOTAL = 0
    FP_TOTAL = 0
    FN_TOTAL = 0

    preds_flat = list()
    gts_flat = list()

    for i in range(len(pred_list)): 

        pred = preds[i]
        gt = gts[i]

        ret, pred = cv2.threshold(pred, thr, 255, cv2.THRESH_BINARY)
        ret, gt = cv2.threshold(gt, thr, 255, cv2.THRESH_BINARY)

        if pred is None:
            print("SOMETHING WENT WRONG with image:")
            print(pred_list[i])

        pred=np.asarray(pred)
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()

        preds_flat.append(pred_flat)
        gts_flat.append(gt_flat)
        
    preds_flat_concat = np.hstack(preds_flat)
    gts_flat_concat = np.hstack(gts_flat)

    print("gt: ",gt_flat.shape)
    print("",pred_flat.shape)
    
    TN, FP, FN, TP = metrics.confusion_matrix(gt_flat,pred_flat).ravel()

    TP_TOTAL += int(TP)
    TN_TOTAL += int(TN)
    FP_TOTAL += int(FP)
    FN_TOTAL += int(FN)
        
    precision = conditional_div(TP_TOTAL, FP_TOTAL+TP_TOTAL)
    recall = conditional_div(TP_TOTAL, TP_TOTAL+FN_TOTAL)
    fallout = conditional_div(FP_TOTAL, FP_TOTAL+TN_TOTAL)
    accuracy = conditional_div(TP_TOTAL+TN_TOTAL, TP_TOTAL+FP_TOTAL+TN_TOTAL+FN_TOTAL)
    f1score = 2*conditional_div(recall*precision, recall+precision)

    precision_list.append(precision)
    recall_list.append(recall)
    fallout_list.append(fallout)
    accuracy_list.append(accuracy)
    f1_list.append(f1score)



thr_best = np.nanargmax(f1_list)

prec_best = precision_list[thr_best]
rec_best = recall_list[thr_best]
fallout_best = fallout_list[thr_best]
acc_best = accuracy_list[thr_best]
f1_best = f1_list[thr_best]

save_path = os.path.join(path_out, "metrics_val")

try:
    os.mkdir(save_path)
except:
    print("")


data = {'Run': [run_name], 'thr': [thr_best], 'acc': [acc_best], 'prec': [prec_best], 'rec': [rec_best], 'fall': [fallout_best], 'f1': [f1_best]} #, 'auc': [roc_auc]}

df = pd.DataFrame(data)
print(df)

df.to_excel(os.path.join(save_path,'metrics.xlsx'))


