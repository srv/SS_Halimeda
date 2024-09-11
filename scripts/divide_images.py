import os
import argparse
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from tqdm import tqdm
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument('--data_in_path', help='Path to the data folder', type=str)
parser.add_argument('--data_out_path', help='Path to the data folder', type=str)
parsed_args = parser.parse_args()

data_in_path = parsed_args.data_in_path
data_out_path = parsed_args.data_out_path

# try:
#     os.mkdir(data_out_path)
# except:
#     print("")

images_PATH = os.path.join(data_in_path, "images")
masks_PATH =  os.path.join(data_in_path, "gt")
out_images_PATH = os.path.join(data_out_path, "images")
out_masks_PATH =  os.path.join(data_out_path, "gt")

images_list = sorted(os.listdir(images_PATH))
masks_t_list = os.listdir(masks_PATH)
masks_list = list()
for im_id in images_list:
    name, extension = im_id.split('.')
    mask_id = name + '_gt'
    indexes = [i for i, _ in enumerate(masks_t_list) if mask_id in masks_t_list[i]]
    masks_list.append(masks_t_list[indexes[0]])

# train images and masks
print('Loading train images') 
for n, id_ in enumerate(images_list):
    path = os.path.join(images_PATH, id_)
    print(path)
    name, extension = id_.split('.')
    img = imread(path)[:, :, :3]
    path = os.path.join(masks_PATH, masks_list[n])
    print(path)
    mask = imread(path)[:, :, :3]

    x_val_1, y_val_1, z_val_1 = img.shape
    # print('Img: X_val = ', str(x_val), '. Y_val = ', str(y_val), '. Z_val = ', str(z_val))
    x_val_2, y_val_2, z_val_2 = mask.shape
    # print('Mask: X_val = ', str(x_val), '. Y_val = ', str(y_val), '. Z_val = ', str(z_val))
    if x_val_1 != x_val_2:
        print('Xval is different: x_val_1 = ', str(x_val_1), '. x_val_2 = ', str(x_val_2))
    if y_val_1 != y_val_2:
        print('yval is different: y_val_1 = ', str(y_val_1), '. y_val_2 = ', str(y_val_2))

    # x_cut = 0
    # y_cut = 0
    # if 1500 > x_val >= 1000:
    #     x_cut = 2
    # elif 2000 > x_val >= 1500:
    #     x_cut = 3
    # elif 2500 > x_val >= 2000:
    #     x_cut = 4
    # elif 3000 > x_val >= 2500:
    #     x_cut = 5
    # elif 3500 > x_val >= 3000:
    #     x_cut = 6
    # elif 4000 > x_val >= 3500:
    #     x_cut = 7
    # elif x_val >= 4000:
    #     x_cut = 8
    
    # if 1500 > y_val >= 1000:
    #     y_cut = 2
    # elif 2000 > y_val >= 1500:
    #     y_cut = 3
    # elif 2500 > y_val >= 2000:
    #     y_cut = 4
    # elif 3000 > y_val >= 2500:
    #     y_cut = 5
    # elif 3500 > y_val >= 3000:
    #     y_cut = 6
    # elif 4000 > y_val >= 3500:
    #     y_cut = 7
    # elif y_val >= 4000:
    #     y_cut = 8

    x_val = x_val_1
    y_val = y_val_1
    x_cut = int(x_val / 1000) if x_val >= 2000 else 0
    y_cut = int(y_val / 1000) if y_val >= 2000 else 0

    #print('X_cut = ', str(x_cut), '. Y_cut = ', str(y_cut))
    if x_cut > 0 and y_cut > 0:
        for i in range(x_cut):
            for j in range(y_cut):
                x_upper_bound = (i+1)*1000 if i < x_cut-1 else x_val-1
                y_upper_bound = (j+1)*1000 if j < y_cut-1 else y_val-1
                print('Image boundaries: X = [', str(i*1000), ', ', str(x_upper_bound), ']. Y = [', str(j*1000), ', ', str(y_upper_bound), ']')
                new_image = img[i*1000:x_upper_bound, j*1000:y_upper_bound, :3]
                imsave(os.path.join(out_images_PATH, name+'_'+str(i)+'_'+str(j)+'.'+extension), new_image)
                new_mask = mask[i*1000:x_upper_bound, j*1000:y_upper_bound, :3]
                imsave(os.path.join(out_masks_PATH, name+'_'+str(i)+'_'+str(j)+'_gt.jpg'), new_mask)
    elif x_cut > 0:
        for i in range(x_cut):
            x_upper_bound = (i+1)*1000 if i < x_cut-1  else x_val-1
            new_image = img[i*1000:x_upper_bound, :, :3]
            imsave(os.path.join(out_images_PATH, name+'_'+str(i)+'.'+extension), new_image)
            new_mask = mask[i*1000:x_upper_bound, :, :3]
            imsave(os.path.join(out_masks_PATH, name+'_'+str(i)+'_gt.jpg'), new_mask)
    elif y_cut > 0:
        for j in range(y_cut):
            y_upper_bound = (j+1)*1000 if j < y_cut-1 else y_val-1
            new_image = img[:, j*1000:y_upper_bound, :3]
            imsave(os.path.join(out_images_PATH, name+'_'+str(j)+'.'+extension), new_image)
            new_mask = mask[:, j*1000:y_upper_bound, :3]
            imsave(os.path.join(out_masks_PATH, name+'_'+str(j)+'_gt.jpg'), new_mask)
    else:
        try:
            new_image = img
            #print('Forth case: ' + str(new_image.shape))
            imsave(os.path.join(out_images_PATH, id_), new_image)
            new_mask = mask
            imsave(os.path.join(out_masks_PATH, name+'_gt.jpg'), new_mask)
        except Exception as e:
            print('Fourth case: ' + str(e))
        
    del img; del mask
