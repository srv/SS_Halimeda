import os
import cv2
import numpy as np
from tqdm import tqdm 
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imread, imshow


IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_CHANNELS = 3

run = "1024_8_default2"
save_path = os.path.join("/home/tintin/SS_Halimeda/runs", run)

TEST_PATH = "/home/tintin/SS_Halimeda/data/orig/test/img"

TEST_PATH = "/home/object/SS_Halimeda/data/"
test_list = sorted(os.listdir(TEST_PATH))


model = tf.keras.models.load_model(os.path.join(save_path, "model.h5"))


# test images
X_test = np.zeros((len(test_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Loading test images') 
for n, id_ in tqdm(enumerate(test_list), total=len(test_list)):
    path = os.path.join(TEST_PATH, id_)
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img


preds_test = model.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

print("INFERENCE DONE")

path_im_out = os.path.join(save_path, "inference_out")
path_thr_out = os.path.join(save_path, "inference_out_t")

try:
    os.mkdir(path_im_out)
    os.mkdir(path_thr_out)
except:
    print("")

for idx, name in enumerate(test_list):
    plt.imsave(os.path.join(path_im_out, name), np.squeeze(preds_test[idx]))
    plt.imsave(os.path.join(path_thr_out, name), np.squeeze(preds_test_t[idx]))

