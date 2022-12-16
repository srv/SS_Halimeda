import os
import cv2
import random
import tensorflow 
import numpy as np
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_out = "/home/uib/Desktop/SS_Halimeda/Halimeda_Images/test_da/out/"
path_in = "/home/uib/Desktop/SS_Halimeda/Halimeda_Images/test_da/in/"

path_out_img= os.path.join(path_out + "img")
path_out_mask= os.path.join(path_out + "mask")

try:
    os.mkdir(path_out_img)
    os.mkdir(path_out_mask)
except:
    print("")

path_in_img=  os.path.join(path_in + "img")
path_in_mask= os.path.join(path_in + "mask")

for image_file in os.listdir(path_in_img):

    s = random.randint(1, 20000)

    img_path = os.path.join(path_in_img, image_file)
    mask_file = os.path.join(path_in_mask,"gt_" + image_file)
    
    image=load_img(img_path)
    mask=load_img(mask_file)
    
    image=np.expand_dims(image,axis=0)
    mask=np.expand_dims(mask,axis=0)

    #datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='reflect')
    datagen = ImageDataGenerator(rotation_range=50,width_shift_range=0.5,height_shift_range=0.5,shear_range=0.5,zoom_range=0.5,horizontal_flip=True,fill_mode='reflect')
        
    aug_iter_img = datagen.flow(image, batch_size=1, seed=s)
    image_aug = next(aug_iter_img)[0].astype('uint8')

    aug_iter_mask = datagen.flow(mask, batch_size=1, seed=s)
    mask_aug = next(aug_iter_mask)[0].astype('uint8')

    save_img(os.path.join(path_out_img,image_file),image_aug)
    save_img(os.path.join(path_out_mask, "gt_" + image_file),mask_aug)



