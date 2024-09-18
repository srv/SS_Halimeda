import os
import random
import shutil
from natsort import natsorted

path_in = "/home/azken/Vicent/Asparagopsis/all/divided_good/all/"
images_PATH = os.path.join(path_in,"images")
masks_PATH = os.path.join(path_in,"gt")

path_out = "/home/azken/Vicent/Asparagopsis/5_fold/"
TRAIN_images_PATH = os.path.join(path_out, "train/images")
TRAIN_masks_PATH = os.path.join(path_out, "train/gt")
VAL_images_PATH = os.path.join(path_out, "val/images")
VAL_masks_PATH = os.path.join(path_out, "val/gt")
TEST_images_PATH = os.path.join(path_out, "test/images")
TEST_masks_PATH = os.path.join(path_out, "test/gt")

n_img = len(os.listdir(images_PATH))

val_split = 0
test_split = 0.1

n_val = int(n_img * val_split)
n_test = int(n_img * test_split)
n_train = n_img - n_val - n_test

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


