import os
import random
import shutil
from natsort import natsorted

path_in = "/home/azken/Vicent/Asparagopsis/5_fold/all_train_val_aug/"
images_PATH = os.path.join(path_in,"images")
masks_PATH = os.path.join(path_in,"gt")

path_out_1 = "/home/azken/Vicent/Asparagopsis/5_fold/cross/1/"
path_out_2 = "/home/azken/Vicent/Asparagopsis/5_fold/cross/2/"
path_out_3 = "/home/azken/Vicent/Asparagopsis/5_fold/cross/3/"
path_out_4 = "/home/azken/Vicent/Asparagopsis/5_fold/cross/4/"
path_out_5 = "/home/azken/Vicent/Asparagopsis/5_fold/cross/5/"

TRAIN_images_PATH_1 = os.path.join(path_out_1, "train/images")
TRAIN_masks_PATH_1 = os.path.join(path_out_1, "train/gt")
VAL_images_PATH_1 = os.path.join(path_out_1, "val/images")
VAL_masks_PATH_1 = os.path.join(path_out_1, "val/gt")

TRAIN_images_PATH_2 = os.path.join(path_out_2, "train/images")
TRAIN_masks_PATH_2 = os.path.join(path_out_2, "train/gt")
VAL_images_PATH_2 = os.path.join(path_out_2, "val/images")
VAL_masks_PATH_2 = os.path.join(path_out_2, "val/gt")

TRAIN_images_PATH_3 = os.path.join(path_out_3, "train/images")
TRAIN_masks_PATH_3 = os.path.join(path_out_3, "train/gt")
VAL_images_PATH_3 = os.path.join(path_out_3, "val/images")
VAL_masks_PATH_3 = os.path.join(path_out_3, "val/gt")

TRAIN_images_PATH_4 = os.path.join(path_out_4, "train/images")
TRAIN_masks_PATH_4 = os.path.join(path_out_4, "train/gt")
VAL_images_PATH_4 = os.path.join(path_out_4, "val/images")
VAL_masks_PATH_4 = os.path.join(path_out_4, "val/gt")

TRAIN_images_PATH_5 = os.path.join(path_out_5, "train/images")
TRAIN_masks_PATH_5 = os.path.join(path_out_5, "train/gt")
VAL_images_PATH_5 = os.path.join(path_out_5, "val/images")
VAL_masks_PATH_5 = os.path.join(path_out_5, "val/gt")

n_img = len(os.listdir(images_PATH))

split = 0.2

n_1 = int(n_img * split)
n_2 = int(n_img * split)
n_3 = int(n_img * split)
n_4 = int(n_img * split)
n_5 = n_1 - n_2 - n_3 - n_4

random_idx = random.sample(range(n_img), (n_img))

split_1_idx = random_idx[:n_1]
split_2_idx = random_idx[n_1:(n_1+n_2)]
split_3_idx = random_idx[(n_1+n_2):(n_1+n_2+n_3)]
split_4_idx = random_idx[(n_1+n_2+n_3):(n_1+n_2+n_3+n_4)]
split_5_idx = random_idx[(n_1+n_2+n_3+n_4):]

img_list = natsorted(os.listdir(images_PATH))
mask_list = natsorted(os.listdir(masks_PATH))

# 1st
for idx in (split_1_idx + split_2_idx + split_3_idx + split_4_idx):
    path_img_from = os.path.join(images_PATH, img_list[idx])
    path_img_to = os.path.join(TRAIN_images_PATH_1, img_list[idx])
    path_mask_from = os.path.join(masks_PATH, mask_list[idx])
    path_mask_to = os.path.join(TRAIN_masks_PATH_1, mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)

for idx in split_5_idx:
    path_img_from = os.path.join(images_PATH, img_list[idx])
    path_img_to = os.path.join(VAL_images_PATH_1, img_list[idx])
    path_mask_from = os.path.join(masks_PATH, mask_list[idx])
    path_mask_to = os.path.join(VAL_masks_PATH_1, mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)

# 2nd
for idx in (split_1_idx + split_2_idx + split_3_idx + split_5_idx):
    path_img_from = os.path.join(images_PATH, img_list[idx])
    path_img_to = os.path.join(TRAIN_images_PATH_2, img_list[idx])
    path_mask_from = os.path.join(masks_PATH, mask_list[idx])
    path_mask_to = os.path.join(TRAIN_masks_PATH_2, mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)

for idx in split_4_idx:
    path_img_from = os.path.join(images_PATH, img_list[idx])
    path_img_to = os.path.join(VAL_images_PATH_2, img_list[idx])
    path_mask_from = os.path.join(masks_PATH, mask_list[idx])
    path_mask_to = os.path.join(VAL_masks_PATH_2, mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)

# 3rd
for idx in (split_1_idx + split_2_idx + split_4_idx + split_5_idx):
    path_img_from = os.path.join(images_PATH, img_list[idx])
    path_img_to = os.path.join(TRAIN_images_PATH_3, img_list[idx])
    path_mask_from = os.path.join(masks_PATH, mask_list[idx])
    path_mask_to = os.path.join(TRAIN_masks_PATH_3, mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)

for idx in split_3_idx:
    path_img_from = os.path.join(images_PATH, img_list[idx])
    path_img_to = os.path.join(VAL_images_PATH_3, img_list[idx])
    path_mask_from = os.path.join(masks_PATH, mask_list[idx])
    path_mask_to = os.path.join(VAL_masks_PATH_3, mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)

# 4th
for idx in (split_1_idx + split_3_idx + split_4_idx + split_5_idx):
    path_img_from = os.path.join(images_PATH, img_list[idx])
    path_img_to = os.path.join(TRAIN_images_PATH_4, img_list[idx])
    path_mask_from = os.path.join(masks_PATH, mask_list[idx])
    path_mask_to = os.path.join(TRAIN_masks_PATH_4, mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)

for idx in split_2_idx:
    path_img_from = os.path.join(images_PATH, img_list[idx])
    path_img_to = os.path.join(VAL_images_PATH_4, img_list[idx])
    path_mask_from = os.path.join(masks_PATH, mask_list[idx])
    path_mask_to = os.path.join(VAL_masks_PATH_4, mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)

# 5th
for idx in (split_2_idx + split_3_idx + split_4_idx + split_5_idx):
    path_img_from = os.path.join(images_PATH, img_list[idx])
    path_img_to = os.path.join(TRAIN_images_PATH_5, img_list[idx])
    path_mask_from = os.path.join(masks_PATH, mask_list[idx])
    path_mask_to = os.path.join(TRAIN_masks_PATH_5, mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)

for idx in split_1_idx:
    path_img_from = os.path.join(images_PATH, img_list[idx])
    path_img_to = os.path.join(VAL_images_PATH_5, img_list[idx])
    path_mask_from = os.path.join(masks_PATH, mask_list[idx])
    path_mask_to = os.path.join(VAL_masks_PATH_5, mask_list[idx])
    shutil.copyfile(path_img_from, path_img_to)
    shutil.copyfile(path_mask_from, path_mask_to)
