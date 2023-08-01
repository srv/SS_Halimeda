import os
import argparse
import numpy as np
# from numba import cuda
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imread, imshow, imsave


"""
Example: 
command = "python3 inference.py --run_path /mnt/c/Users/haddo/SS_Halimeda/model/ --data_path /mnt/c/Users/haddo/Halimeda/merged_model_0/val --shape 1024 "

"""

parser = argparse.ArgumentParser()
parser.add_argument('--run_path', help='Path to the run folder', type=str)
parser.add_argument('--data_path', help='Path to the data folder', type=str)
parser.add_argument('--shape', help='img_shape', type=int)
parser.add_argument('--shape_out', default = 0, help='img_shape_out', type=int)
parsed_args = parser.parse_args()

run_path = parsed_args.run_path
data_path = parsed_args.data_path
shape = parsed_args.shape
shape_out = parsed_args.shape_out

IMG_WIDTH = shape
IMG_HEIGHT = shape
IMG_CHANNELS = 3

save_path = os.path.join(run_path, "inference")

try:
    os.mkdir(save_path)
except:
    print("")

TEST_PATH = data_path

test_list = sorted(os.listdir(TEST_PATH))

X_test = np.zeros((len(test_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
print('Loading test images') 
for n, id_ in enumerate(test_list):
    path = os.path.join(TEST_PATH, id_)
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

# device = cuda.get_current_device()
# device.reset()
model = tf.keras.models.load_model(os.path.join(run_path, "model.h5"))

print("Starting inference")
preds_test = model.predict(X_test, verbose=1)
print("INFERENCE DONE")


for idx, name in enumerate(test_list):

    base, ext = os.path.splitext(name)
    #imsave(os.path.join(save_path, name), np.squeeze(X_test[idx]))
    img = np.squeeze(preds_test[idx])
    if shape_out != 0:
        img = resize(img, (shape_out, shape_out), mode='constant', preserve_range=True)


    imsave(os.path.join(save_path, base + "_grey" + ext), img)

