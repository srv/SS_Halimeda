import os
import cv2
import numpy as np
from numba import cuda
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imread, imshow, imsave

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

run = "1024_8_default2"
save_path = os.path.join("/home/object/SS_Halimeda/runs", run)

TEST_PATH = "/home/object/SS_Halimeda/data/splits/base/test/img"

test_list = sorted(os.listdir(TEST_PATH))


model = tf.keras.models.load_model(os.path.join(save_path, "model.h5"))

X_test = np.zeros((len(test_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
print('Loading test images') 
for n, id_ in enumerate(test_list):
    path = os.path.join(TEST_PATH, id_)
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

#device = cuda.get_current_device()
#device.reset()

print("Starting inference")
preds_test = model.predict(X_test, verbose=1)
print("INFERENCE DONE")

path_im_out = os.path.join(save_path, "inference")

try:
    os.mkdir(path_im_out)
except:
    print("")

for idx, name in enumerate(test_list):

    base, ext = os.path.splitext(name)
    imsave(os.path.join(path_im_out, name), np.squeeze(X_test[idx]))
    imsave(os.path.join(path_im_out, base + "_grey" + ext), np.squeeze(preds_test[idx]))

