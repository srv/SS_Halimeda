import os
import cv2
import random
import numpy as np
from tqdm import tqdm 
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, imshow

 
# read the image file
img1 = cv2.imread(r"C:\Users\person\FDR_Images\IMGsCM\Case3\img1.png",2)
img2 = cv2.imread(r"C:\Users\person\FDR_Images\IMGsCM\Case3\img2.png",2)

ret, bw_img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
ret, bw_img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary1", bw_img1)
cv2.imshow("Binary2", bw_img2)

rows1,cols1 = bw_img1.shape
rows2,cols2 = bw_img2.shape

confMatrix=np.zeros((2,2))

for i in range(rows1):
    for j in range(cols1):
       if bw_img1[i,j]==0:
           if bw_img2[i,j]==0:
               confMatrix[1,1]=confMatrix[1,1]+1
           else:
               confMatrix[0,1]=confMatrix[0,1]+1
       else:
           if bw_img2[i,j]==0:
               confMatrix[1,0]=confMatrix[1,0]+1
           else:
               confMatrix[0,0]=confMatrix[0,0]+1
               
print(confMatrix)
                  
ax = sns.heatmap(confMatrix, annot=True, cmap='Blues', fmt='g')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nActual Values')
ax.set_ylabel('Predicted Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

## Display the visualization of the Confusion Matrix.
plt.show()
    
TP=confMatrix[0,0]
TN=confMatrix[1,1]
FP=confMatrix[1,0]
FN=confMatrix[0,1]
           
Precision = TP / (FP + TP)
Recall = TP / (TP + FN)
F1score = 2*(Recall*Precision)/(Recall + Precision)
Accuracy = (TP + TN)/(TP + FP + TN + FN)

print('Precision',Precision)
print('Recall',Recall)
print('F1score',F1score)
print('Accuracy',Accuracy)
print('TP:',TP)
print('TN:',TN)
print('FP:',FP)
print('FN:',FN)


cv2.waitKey(0)
cv2.destroyAllWindows()