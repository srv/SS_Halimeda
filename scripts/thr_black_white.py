import imageio
from scipy import ndimage
import scipy
import os
import re
import numpy as np
import random
import shutil
from tqdm import tqdm 
from skimage.io import imread, imshow
seed = 42
np.random.seed = seed
save_path="/home/uib/Documentos/cat/INVHALI/data_to_hd/halimeda/sets/semantic/cabrera_512/"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

"""
binarize an image
"""

# INITIALIZATIONS:

TRAIN_images_PATH = save_path+"/images/"
TRAIN_masks_PATH = save_path+"/masks/"
TRAIN_masks_PATH_thr=save_path+"/masks_thr/"
TEST_FOLDER= save_path+"/test/"
TRAIN_FOLDER=save_path+"/train/"

not_binarized_yet=True

if not os.path.exists(TEST_FOLDER):
      os.mkdir(TEST_FOLDER)

if not os.path.exists(TRAIN_FOLDER):
      os.mkdir(TRAIN_FOLDER)

if not os.path.exists(TRAIN_masks_PATH_thr):
    os.mkdir(TRAIN_masks_PATH_thr)

#PART 1: BINARIZE MASKS

if not_binarized_yet:
    for image_file in os.listdir(TRAIN_masks_PATH):  # for each file in the folder
        #   print(image_file)
        image = imageio.imread(TRAIN_masks_PATH+image_file) # read image
        img = np.full([image.shape[0], image.shape[1]], 0, dtype=np.uint8)  # auxiliary image

        thr = 127   # set threshold
        y = np.where(image > thr)  # above threshold
        img[y[0], y[1]] = 255  # set to white

        y = np.where(image <= thr)  # below threshold
        img[y[0], y[1]] = 0  # set to black
        # print(set(img.flatten()))
        image_id=image_file.split(".")[0]
        # print(image_file)
        imageio.imsave(TRAIN_masks_PATH_thr + "/" + image_id+".png", img)  # generate image file

#PART 2: CREATE TEST SPLIT
TEST_SPLIT=0.1
test_list=[]
test_masks=[]
train_images_list = sorted(os.listdir(TRAIN_images_PATH))
train_masks_list = sorted(os.listdir(TRAIN_masks_PATH_thr))
num_images=len(train_images_list)

print("NUM IMAGES:",num_images)
#check:
if  num_images != len(train_masks_list):
    print("WARNING: THE NUMBER OF MASKS AND IMAGES IS DIFFER!")

#select test set randomly
test_set_len=int(TEST_SPLIT*len(train_images_list))
for i in range(test_set_len):
    idx=random.randint(0, len(train_images_list)-1)
    test_list.append(train_images_list[idx])
    test_masks.append(train_masks_list[idx])
    train_images_list.pop(idx)
    train_masks_list.pop(idx)

if not os.path.exists(TEST_FOLDER+"images"):
  os.mkdir(TEST_FOLDER+"images")
if not os.path.exists(TEST_FOLDER+"masks"):
  os.mkdir(TEST_FOLDER+"masks")

if not os.path.exists(TRAIN_FOLDER+"images"):
      os.mkdir(TRAIN_FOLDER+"images")
if not os.path.exists(TRAIN_FOLDER+"masks"):
  os.mkdir(TRAIN_FOLDER+"masks")


#CHECK!
set_imgs=[]
set_imgs.extend(test_list)
print("len test_list is ",len(test_list))
print("len set ",len(set(set_imgs)))

set_imgs.extend(train_images_list)
print("len_train_list is ",len(train_images_list))
print("len set ",len(set(set_imgs)))

set_imgs.extend(train_masks_list)
print("train_mask_list is ",len(train_masks_list))
print("len set ",len(set(set_imgs)))

for image,mask in zip(test_list,test_masks):
    shutil.copyfile(TRAIN_images_PATH+"/"+image,TEST_FOLDER+"/images/"+image)
    shutil.copyfile(TRAIN_masks_PATH_thr+"/"+mask,TEST_FOLDER+"/masks/"+mask)

for image,mask in zip(train_images_list,train_masks_list):
    shutil.copyfile(TRAIN_images_PATH+"/"+image,TRAIN_FOLDER+"/images/"+image)
    shutil.copyfile(TRAIN_masks_PATH_thr+"/"+mask,TRAIN_FOLDER+"/masks/"+mask)

#check!
# save_path="/content/drive/MyDrive/HALIMEDA"
print(train_images_list[0:10])
print(train_masks_list[0:10])
print("")
print(train_images_list[-10:])
print(train_masks_list[-10:])
print("")
print(test_list[0:10])
print(test_masks[0:10])
print("")
print(test_list[-10:])
print(test_masks[-10:])

#CREATE AND SAVE data vectors

new_train_imgs_list=sorted(train_images_list)
new_train_masks_list=sorted(train_masks_list)
new_test_list=sorted(test_list)
new_test_masks_list=sorted(test_masks)

if new_train_imgs_list != train_images_list:
    print("PROBLEM FOUND 1!!!")

if new_train_masks_list != train_masks_list:
    print("PROBLEM FOUND 2!!!")
    
if new_test_list != test_list:
    for idx,elem in enumerate(new_test_list):
        if elem!=test_list[idx]:
            print("PROBLEM IN ELEM :",idx)
            print("new_test_list :",new_test_list[idx])
            print("test_list :",test_list[idx])
    print("PROBLEM FOUND 3!!!")
    
if new_test_masks_list != test_masks:
    for idx,elem in enumerate(new_test_list):
        if elem!=test_list[idx]:
            print("PROBLEM IN ELEM :",idx)
            print("new_test_list :",new_test_list[idx])
            print("test_list :",test_list[idx])
    print("PROBLEM FOUND 4 !!!")


train_images_list = new_train_imgs_list
train_masks_list = new_train_masks_list
test_list = new_test_list
test_masks = new_test_masks_list

llista=[]
# train images
X_train = np.zeros((len(train_images_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
print('Resizing train images') 
for n, id_ in tqdm(enumerate(train_images_list), total=len(train_images_list)):
    path = TRAIN_images_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img

# masks images
Y_train = np.zeros((len(train_masks_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
print('Resizing masks images') 
for n, id_ in tqdm(enumerate(train_masks_list), total=len(train_masks_list)):
    path = TRAIN_masks_PATH_thr + id_
    mask = imread(path)[:,:]
    mask_exp=np.expand_dims(mask,axis=2)
    # mask = (resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True))
    Y_train[n] = mask_exp
    llista.extend(mask)

# test images
X_test = np.zeros((len(test_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
TEST_PATH=TEST_FOLDER+"/images/"
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_list), total=len(test_list)):
    path = TEST_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)

np.save(save_path+"/Xtrain_C512",X_train)
np.save(save_path+"/Ytrain_C512",Y_train)
np.save(save_path+"/Xtest_C512",X_test)



new_list=[int(elem[0]) for elem in llista]
print(set(new_list))

test=np.load(save_path+"/Xtrain_C512.npy",allow_pickle=True)
print(test.shape)