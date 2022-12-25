import os
import random
import shutil
from natsort import natsorted

path_in="/home/tintin/SS_Halimeda/data/1024_1024"
images_PATH = os.path.join(path_in,"img")
masks_PATH = os.path.join(path_in,"mask")

path_out="/home/tintin/SS_Halimeda/data/splits/base"
TRAIN_images_PATH = os.path.join(path_out,"train/img")
TRAIN_masks_PATH = os.path.join(path_out,"train/mask")
VAL_images_PATH = os.path.join(path_out,"val/img")
VAL_masks_PATH = os.path.join(path_out,"val/mask")
TEST_images_PATH = os.path.join(path_out,"test/img")
TEST_masks_PATH = os.path.join(path_out,"test/mask")

n_img = len(os.listdir(images_PATH))

val_split = 0.1
test_split = 0.1

n_val = int(n_img * val_split)
n_test = int(n_img * test_split)
n_train = n_img-n_val-n_test

random_idx = random.sample(range(n_img), (n_img))

train_idx = random_idx[:n_train]
val_idx = random_idx[n_train:(n_train+n_val)]
test_idx = random_idx[(n_train+n_val):]

img_list = natsorted(os.listdir(images_PATH))
mask_list = natsorted(os.listdir(masks_PATH))

for idx in train_idx:
    path_img_from = os.path.join(images_PATH,img_list[idx])
    path_img_to = os.path.join(TRAIN_images_PATH,img_list[idx])
    path_mask_from = os.path.join(masks_PATH,mask_list[idx])
    path_mask_to = os.path.join(TRAIN_masks_PATH,mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)

for idx in val_idx:
    path_img_from = os.path.join(images_PATH,img_list[idx])
    path_img_to = os.path.join(VAL_images_PATH,img_list[idx])
    path_mask_from = os.path.join(masks_PATH,mask_list[idx])
    path_mask_to = os.path.join(VAL_masks_PATH,mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)


for idx in test_idx:
    path_img_from = os.path.join(images_PATH,img_list[idx])
    path_img_to = os.path.join(TEST_images_PATH,img_list[idx])
    path_mask_from = os.path.join(masks_PATH,mask_list[idx])
    path_mask_to = os.path.join(TEST_masks_PATH,mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)
    print(path_mask_from)
    print(path_mask_to)


