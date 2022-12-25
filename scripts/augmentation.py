import os
import cv2
import random
import tensorflow 
import numpy as np
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_in = "/home/tintin/SS_Halimeda/data/splits/da/train/"
path_out = "/home/tintin/SS_Halimeda/data/splits/da/train_da/"


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

    name, ext = os.path.splitext(image_file)
    img_path = os.path.join(path_in_img, image_file)
    mask_file = os.path.join(path_in_mask, name + "_gt" + ext)
    
    image=load_img(img_path)
    mask=load_img(mask_file)
    
    image=np.expand_dims(image,axis=0)
    mask=np.expand_dims(mask,axis=0)

    datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.3,zoom_range=0.25,horizontal_flip=True,fill_mode='reflect')
    #datagen = ImageDataGenerator(rotation_range=50,width_shift_range=0.5,height_shift_range=0.5,shear_range=0.5,zoom_range=0.5,horizontal_flip=True,fill_mode='reflect')
        
    aug_iter_img = datagen.flow(image, batch_size=1, seed=s)
    image_aug = next(aug_iter_img)[0].astype('uint8')

    aug_iter_mask = datagen.flow(mask, batch_size=1, seed=s)
    mask_aug = next(aug_iter_mask)[0].astype('uint8')

    save_img(os.path.join(path_out_img, name + "_da" + ext), image_aug)
    save_img(os.path.join(path_out_mask, name + "_da_gt" + ext), mask_aug)



