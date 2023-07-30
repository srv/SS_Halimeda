import os
import cv2
import random
import tensorflow 
import numpy as np
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_in = "/home/olivia/Halimeda/semantic_segmentation/SS_Halimeda/data/1/train"
path_out = "/home/olivia/Halimeda/semantic_segmentation/SS_Halimeda/data_da/1_da/train"


path_out_img= os.path.join(path_out, "img")
path_out_mask= os.path.join(path_out, "mask")

try:
    os.mkdir(path_out_img)
    os.mkdir(path_out_mask)
except:
    print("")

path_in_img=  os.path.join(path_in, "img")
path_in_mask= os.path.join(path_in, "mask")

for image_file in os.listdir(path_in_img):

    s = random.randint(1, 20000)

    name, ext = os.path.splitext(image_file)
    img_path = os.path.join(path_in_img, image_file)
    mask_file = os.path.join(path_in_mask, name + "_gt" + ext)
    
    image=load_img(img_path)
    mask=load_img(mask_file)
    
    image=np.expand_dims(image,axis=0)
    mask=np.expand_dims(mask,axis=0)

    datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.15,zoom_range=0.15,horizontal_flip=True,fill_mode='reflect')
    #datagen = ImageDataGenerator(rotation_range=50,width_shift_range=0.5,height_shift_range=0.5,shear_range=0.5,zoom_range=0.5,horizontal_flip=True,fill_mode='reflect')
    
    a = random.randint(0, 4)	
    if a < 4:
        aug_iter_img = datagen.flow(image, batch_size=1, seed=s)
        image = next(aug_iter_img)[0].astype('uint8')

        aug_iter_mask = datagen.flow(mask, batch_size=1, seed=s)
        mask = next(aug_iter_mask)[0].astype('uint8')

    else:
        image = image.astype('uint8')
        mask = mask.astype('uint8')
        image = image[0,:,:,:]
        mask = mask[0,:,:,:]

    save_img(os.path.join(path_out_img, name + "_da" + ext), image)
    save_img(os.path.join(path_out_mask, name + "_da_gt" + ext), mask)



