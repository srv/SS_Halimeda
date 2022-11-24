# %%
import numpy as np
import tensorflow as tf
import os
import random
import cv2 #pip install opencv-python
 
from tqdm import tqdm 
from numba import cuda

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import u_net

seed = 42
np.random.seed = seed

save_path="model_outputs"

IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_CHANNELS = 3

base="/home/plome/DATA/INVHALI/sets/semantic/c_c_leia_bckgrnds"
TRAIN_images_PATH = base + "/images/"
TRAIN_masks_PATH = base + "/masks/"

TEST_SPLIT=0.1
test_list=[]

train_images_list = sorted(os.listdir(TRAIN_images_PATH))
train_masks_list = sorted(os.listdir(TRAIN_masks_PATH))
num_images=len(train_images_list)

print("NUM IMAGES:",num_images)
#check:
if  num_images != len(train_masks_list):
    print("WARNING: THE NUMBER OF MASKS AND IMAGES IS DIFFER!")


#select test set randomly
test_set_len=TEST_SPLIT*len(train_images_list)
for i in range(test_set_len):

    idx=randint(0, len(train_images_list)-1)
    test_list.append(train_images_list[idx])
    train_images_list.pop(idx)
    train_masks_list.pop(idx)


set_imgs=[]
set_imgs.extend(test_list)
print("len test_list is ",len(test_list))
print("len set ",len(set(set_imgs)))

set_imgs.extend(train_list)
print("len_train_list is ",train_list)
print("len set ",len(set(set_imgs)))

set_imgs.extend(train_masks_list)
print("train_mask_list is ",train_masks_list)
print("len set ",len(set(set_imgs)))



# %%

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
    path = TRAIN_masks_PATH + id_
    mask = imread(path)[:,:,:1]
    # mask = (resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True))
    Y_train[n] = mask

# test images
X_test = np.zeros((len(test_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_list), total=len(test_list)):
    path = TEST_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    # img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img


np.save(save_path+"/Xtrain1024_L",X_train)
np.save(save_path+"/Ytrain1024_L",Y_train)
np.save(save_path+"/Xtest1024_L",X_test)

# %%
X_train=np.load(save_path+"/Xtrain1024_L.npy",allow_pickle=True)
Y_train=np.load(save_path+"/Ytrain1024_L.npy",allow_pickle=True)
X_test=np.load(save_path+"/Xtest1024_L.npy",allow_pickle=True)


print('Done!')
print('Xtrain:',X_train.shape)
print('Ytrain:',Y_train.shape)
print('Xtest:',X_test.shape)

plt.figure()
imshow(X_train[0])

plt.figure()
imshow(Y_train[0])

# %%
device = cuda.get_current_device()
device.reset()

#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
outputs = tf.keras.layers.Conv2D(1,(1,1), activation='sigmoid')(c9) # binary activation output
#outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
# opt = tf.keras.optimizers.Adam(learning_rate=1e-6)
# model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy', 'mse', 'mae', 'mape','Precision','Recall'])
model.compile(optimizer= 'Adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', 'mae', 'mape','Precision','Recall'])

model.summary()

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('halimeda_SS_L.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=300, callbacks=callbacks)

print("END OF TRAINING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

tf.keras.models.save_model(model,save_path+"/Halimeda_SS.h5")
print("model saved!!!")




####################################
#plot metrics

plt.figure()
plt.plot(results.history['mse'])
plt.title('mse')
plt.savefig(save_path+"/mse")
#plt.show()

plt.plot(results.history['mae'])
plt.title('mae')
plt.figure()
plt.savefig(save_path+"/mae")
#plt.show()

plt.plot(results.history['mape'])
plt.title('mape')
plt.figure()
plt.savefig(save_path+"/mape")
#plt.show()

plt.plot(results.history['loss'])#train_loss
plt.title('train_loss')
plt.figure()
plt.savefig(save_path+"/train_loss")
#plt.show()

plt.plot(results.history['val_loss'])
plt.title('val_loss')
plt.figure()
plt.savefig(save_path+"/val_loss")
#plt.show()

plt.plot(results.history['accuracy'])
plt.title('train_accuracy')
plt.figure()
plt.savefig(save_path+"/train_accuracy")
#plt.show()

plt.plot(results.history['val_accuracy'])
plt.title('val_accuracy')
plt.figure()
plt.savefig(save_path+"/val_acuracy")
#plt.show()

#comparisons
plt.plot(results.history['loss'])#train_loss
plt.plot(results.history['val_loss'])
plt.title('train_loss & val_loss')
plt.figure()
plt.savefig(save_path+"/train_val_loss")
#plt.show()

plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('train_accuracy & val_accuracy')
plt.figure()
plt.savefig(save_path+"/train_val_accuracy")
#plt.show()

plt.figure()
plt.plot(results.history['precission'])
plt.plot(results.history['recall'])
plt.title('Precission and recall')
plt.figure()
plt.savefig(save_path+"/Precission_and_recall")


####################################


idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

print("PREDICTIONS DONE!!!!!!!!!!!!!!!!!!!")

np.save(save_path+"/preds_train",preds_train)
np.save(save_path+"/preds_val",preds_val)
np.save(save_path+"/preds_test",preds_test)
#thr
np.save(save_path+"/preds_train_t",preds_train_t)
np.save(save_path+"/preds_val_t",preds_val_t)
np.save(save_path+"/preds_test_t",preds_test_t)

print(type(results.history))
np.save(save_path+"/history",results.history)
np.save(save_path+"/results",results)


# %%
# %%
print(results.history.keys())
# %%
for idx, name in enumerate(test_list):
    plt.imsave(save_path+"/inference_out/" + name, np.squeeze(preds_test[idx]))
    plt.imsave(save_path+"/inference_out_t/" + name, np.squeeze(preds_test_t[idx]))


# ####!tensorboard --logdir=logs/ --host localhost --port 8088

# %%

# INFERENCE


save_path="model_outputs"

IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_CHANNELS = 3

INFER_PATH = "/home/object/SS_Halimeda/Halimeda_Images/SubSelection_GoodFor_Segmentation/DetailedIMGs/halimeda_train/images/"
infer_list = sorted(os.listdir(INFER_PATH))


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
    plt.imsave(save_path+"/inference_out_leia_train/" + name, np.squeeze(preds_infer[idx]))
    plt.imsave(save_path+"/inference_out_t_leia_train/" + name, np.squeeze(preds_infer_t[idx]))


# %%
