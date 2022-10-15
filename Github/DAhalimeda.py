import tensorflow 
import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img






DG_folder='C:/Documents/TFG/TFG_Images/IMGs_SelectionOfImages/SubSelection_GoodFor_Segmentation/DA_NotDetailedIMGs/halimeda_train/images'
DG_folder1='C:/Documents/TFG/TFG_Images/IMGs_SelectionOfImages/SubSelection_GoodFor_Segmentation/DA_NotDetailedIMGs/halimeda_train/masks'
images_increased = 5

try:
    os.mkdir(DG_folder)
    os.mkdir(DG_folder1)
except:
    print("")


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect')


data_path = "C:/Users/tomas/Documents/TFG/TFG_Images/IMGs_SelectionOfImages/SubSelection_GoodFor_Segmentation/NotDetailedIMGs/halimeda_train/images" 
mask_path = "C:/Users/tomas/Documents/TFG/TFG_Images/IMGs_SelectionOfImages/SubSelection_GoodFor_Segmentation/NotDetailedIMGs/halimeda_train/masks" 

data_dir_list = os.listdir(data_path)


width_shape, height_shape = 1024,1024

i=0
num_images=0
for image_file in data_dir_list:
    img_list=os.listdir(data_path)
    print(image_file)

    img_path = data_path + '/'+ image_file
    mask_file = mask_path + '/gt_'+image_file
    
    imge=load_img(img_path)
    mask=load_img(mask_file)
    
    imge=cv2.resize(tensorflow.keras.utils.img_to_array(imge), (width_shape, height_shape), interpolation = cv2.INTER_AREA)
    mask=cv2.resize(tensorflow.keras.utils.img_to_array(mask), (width_shape, height_shape), interpolation = cv2.INTER_AREA)

    x= imge/255
    y= mask/255

    x=np.expand_dims(x,axis=0)
    y=np.expand_dims(y,axis=0)
    t=1
    for output_batch_img, output_batch_mask in zip(train_datagen.flow(x,batch_size=1), train_datagen.flow(y,batch_size=1)):
        a=tensorflow.keras.utils.img_to_array(output_batch_img[0])
        imagen=output_batch_img[0,:,:]*255
        imgfinal = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        cv2.imwrite(DG_folder+"/%i%i.jpg"%(i,t), imgfinal)
        
        b=tensorflow.keras.utils.img_to_array(output_batch_mask[0])
        imagen1=output_batch_mask[0,:,:]*255
        imgfinal1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2RGB)
        cv2.imwrite(DG_folder1+"/gt_%i%i.jpg"%(i,t), imgfinal1)
        t+=1
        
        num_images+=1
        if t>images_increased:
            break
    i+=1
    
print("images generated",num_images)
