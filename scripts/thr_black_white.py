# %% 

import imageio.v2 as imageio
from scipy import ndimage
import scipy
import os
import re
import numpy as np
import random
import shutil
from tqdm import tqdm 
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize
# from numba import cuda
import u_net
seed = 42
np.random.seed = seed

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

batch_size=8
lr=1e-3

base_path="/home/plome/DATA/INVHALI/sets/semantic/c_c_leia_bckgrnds"
out_path="/home/plome/SS_Halimeda/entrenos/c_c_leia_bckgrnds"

TRAIN_images_PATH = base_path+"/images/"
TRAIN_masks_PATH = base_path+"/masks/"

TRAIN_masks_PATH_thr=out_path+"/masks_thr/"
TEST_FOLDER= out_path+"/test/"
TRAIN_FOLDER=out_path+"/train/"

model_name="halimeda_SS_C_C_L_b512.h5"
model_suffix="_C_C_L_b512"

# INITIALIZATIONS:
# %% 

not_binarized_yet=True

if not os.path.exists(TEST_FOLDER):
      os.mkdir(TEST_FOLDER)

if not os.path.exists(TRAIN_FOLDER):
      os.mkdir(TRAIN_FOLDER)

if not os.path.exists(TRAIN_masks_PATH_thr):
    os.mkdir(TRAIN_masks_PATH_thr)

#PART 1: BINARIZE MASKS
print(len(os.listdir(TRAIN_masks_PATH)))

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

print(len(os.listdir(TRAIN_masks_PATH_thr)))



#%%
#PART 2: CREATE TEST SPLIT
TEST_SPLIT=0.1
test_list=[]
test_masks=[]
train_images_list = sorted(os.listdir(TRAIN_images_PATH))
train_masks_list = sorted(os.listdir(TRAIN_masks_PATH_thr))
num_images=len(train_images_list)

print("NUM IMAGES:",num_images)
#%%
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

#CREATE AND SAVE data vectors sorted (this lines can be deleted)

new_train_imgs_list=sorted(train_images_list)
new_train_masks_list=sorted(train_masks_list)
new_test_list=sorted(test_list)
new_test_masks_list=sorted(test_masks)

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

np.save(out_path+"/Xtrain"+model_suffix+".npy",X_train)
np.save(out_path+"/Ytrain"+model_suffix+".npy",Y_train)
np.save(out_path+"/Xtest"+model_suffix+".npy",X_test)

new_list=[int(elem[0]) for elem in llista]
print(set(new_list))

## THRESHOLDING AND SPLIT DONE!!----------------------------------------------------

# %%
# X_train=np.load(out_path+"/Xtrain_C_C_L_b512.npy",allow_pickle=True)
# Y_train=np.load(out_path+"/Ytrain_C_C_L_b512.npy",allow_pickle=True)
# X_test=np.load(out_path+"/Xtest_C_C_L_b512.npy",allow_pickle=True)

#CHECK!!

print('Done!')
print('Xtrain:',X_train.shape)
print('Ytrain:',Y_train.shape)
print('Xtest:',X_test.shape)
i=random.randint(0,X_train.shape[0]-1)

train_images_list[i]
print("IMAGE IS : ",train_images_list[i])

print()
print("ploting ",i," th image")
plt.figure()
imshow(X_train[i])

plt.figure()
imshow(Y_train[i])

# %%
#BUILD MODEL AND TRAIN:

# device = cuda.get_current_device()
# device.reset()

#Build the model
model=u_net.define_unet(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)

f = open(out_path+"params.txt", "a")
f.write("Params used in this training:")
f.write("batch size: "+ str(batch_size))
f.write("lr: "+ str(lr))
f.close()

# opt = tf.keras.optimizers.Adam(learning_rate=lr)
# model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy', 'mse', 'mae', 'mape','Precision','Recall'])

model.compile(optimizer= 'Adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', 'mae', 'mape','Precision','Recall'])

model.summary()

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint(model_name, verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=batch_size, epochs=300, callbacks=callbacks)

print("END OF TRAINING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# %%
tf.keras.models.save_model(model,out_path+"/"+model_name)
print("model saved!!!")


####################################
#plot metrics

plt.figure()
plt.plot(results.history['mse'])
plt.title('mse')
plt.savefig(out_path+"/mse")

#plt.show()
plt.figure()
plt.plot(results.history['mae'])
plt.title('mae')
plt.savefig(out_path+"/mae")
#plt.show()
plt.figure()
plt.plot(results.history['mape'])
plt.title('mape')
plt.savefig(out_path+"/mape")


#plt.show()
plt.figure()
plt.plot(results.history['loss'])#train_loss
plt.title('train_loss')
plt.savefig(out_path+"/train_loss")
#plt.show()

plt.figure()
plt.plot(results.history['val_loss'])
plt.title('val_loss')
plt.savefig(out_path+"/val_loss")
#plt.show()

plt.figure()
plt.plot(results.history['accuracy'])
plt.title('train_accuracy')
plt.savefig(out_path+"/train_accuracy")
#plt.show()

plt.figure()
plt.plot(results.history['val_accuracy'])
plt.title('val_accuracy')
plt.savefig(out_path+"/val_acuracy")
#plt.show()

#comparisons
plt.figure()
plt.plot(results.history['loss'])#train_loss
plt.plot(results.history['val_loss'])
plt.title('train_loss & val_loss')
plt.savefig(out_path+"/train_val_loss")
#plt.show()
plt.figure()
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('train_accuracy & val_accuracy')
plt.savefig(out_path+"/train_val_accuracy")
#plt.show()

plt.figure()
plt.plot(results.history['precision'])
plt.plot(results.history['recall'])
plt.title('Precission and recall')
plt.savefig(out_path+"/Precision_and_recall")


####################################

# %%
idx = random.randint(0, len(X_train)-1)

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

print("PREDICTIONS DONE!!!!!!!!!!!!!!!!!!!")

np.save(out_path+"/preds_train",preds_train)
np.save(out_path+"/preds_val",preds_val)
np.save(out_path+"/preds_test",preds_test)
#thr
np.save(out_path+"/preds_train_t",preds_train_t)
np.save(out_path+"/preds_val_t",preds_val_t)
np.save(out_path+"/preds_test_t",preds_test_t)

print(type(results.history))
np.save(out_path+"/history",results.history)
np.save(out_path+"/results",results)


# %%

if not os.path.exists(out_path+"/inference_out/"):

    os.mkdir(out_path+"/inference_out/")
    os.mkdir(out_path+"/inference_out_t/")



for idx, name in enumerate(test_list):
    plt.imsave(out_path+"/inference_out/" + name, np.squeeze(preds_test[idx]))
    plt.imsave(out_path+"/inference_out_t/" + name, np.squeeze(preds_test_t[idx]))


# ####!tensorboard --logdir=logs/ --host localhost --port 8088

# %%

# INFERENCE
# IMG_WIDTH = 512
# IMG_HEIGHT = 512
# IMG_CHANNELS = 3

save_path = out_path

INFER_PATH = "/home/plome/SS_Halimeda/entrenos/c_c_leia_bckgrnds/test/images/"

infer_list = sorted(os.listdir(INFER_PATH))


if not os.path.exists(save_path+"/inference_out/"):

    os.mkdir(save_path+"/inference_out/")
    os.mkdir(save_path+"/inference_out_t/")

# infer images
X_infer = np.zeros((len(infer_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_infer = []
print('Resizing infer images') 
for n, id_ in tqdm(enumerate(infer_list), total=len(infer_list)):
    path = INFER_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_infer.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_infer[n] = img


preds_infer = model.predict(X_infer, verbose=1)
preds_infer_t = (preds_infer > 0.5).astype(np.uint8)
for idx, name in enumerate(infer_list):
    plt.imsave(save_path+"/inference_out/" + name, np.squeeze(preds_infer[idx]))
    plt.imsave(save_path+"/inference_out_t/" + name, np.squeeze(preds_infer_t[idx]))


# %%


