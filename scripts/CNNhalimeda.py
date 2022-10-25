import numpy as np
import tensorflow as tf
import os
import random
import cv2 #pip install opencv-python
 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

seed = 42
np.random.seed = seed

save_path="model_outputs"

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_images_PATH = "/home/object/SS_Halimeda/Halimeda_Images/SubSelection_GoodFor_Segmentation/DetailedIMGs/halimeda_train/images/"
TRAIN_masks_PATH = "/home/object/SS_Halimeda/Halimeda_Images/SubSelection_GoodFor_Segmentation/DetailedIMGs/halimeda_train/masks/"
TEST_PATH = "/home/object/SS_Halimeda/Halimeda_Images/SubSelection_GoodFor_Segmentation/DetailedIMGs/halimeda_test/"

train_images_list = os.listdir(TRAIN_images_PATH)
train_masks_list = os.listdir(TRAIN_masks_PATH)
test_list = os.listdir(TEST_PATH)



# train images
X_train = np.zeros((len(train_images_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
#sizes_test = []
print('Resizing train images') 
for n, id_ in tqdm(enumerate(train_images_list), total=len(train_images_list)):
    path = TRAIN_images_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    #sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img

# masks images
Y_train = np.zeros((len(train_masks_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
#sizes_test = []
print('Resizing masks images') 
for n, id_ in tqdm(enumerate(train_masks_list), total=len(train_masks_list)):
    path = TRAIN_masks_PATH + id_
    mask = imread(path)[:,:,:1]
    mask = (resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True))
    Y_train[n] = mask

    
# test images
X_test = np.zeros((len(test_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_list), total=len(test_list)):
    path = TEST_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')
print('Xtrain:',X_train.shape)
print('Ytrain:',Y_train.shape)
print('Xtest:',X_test.shape)



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
opt = tf.keras.optimizers.Adam(learning_rate=1e-6)
model.compile(optimizer= opt, loss='binary_crossentropy', metrics=['accuracy', 'mse', 'mae', 'mape','Precision','Recall'])
#model.compile(optimizer= 'Adam', loss='binary_crossentropy', metrics=['accuracy', 'mse', 'mae', 'mape'])
model.summary()

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('halimeda_SS.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=2, epochs=100, callbacks=callbacks)

tf.keras.models.save_model(model,save_path+"/Halimeda_SS.h5")

####################################
#plot metrics
plt.figure()
plt.plot(results.history['mse'])
plt.title('mse')
plt.figure()
plt.savefig(save_path)


#plt.show()
plt.plot(results.history['mae'])
plt.title('mae')
plt.figure()
plt.savefig(save_path)

#plt.show()
plt.plot(results.history['mape'])
plt.title('mape')
plt.figure()
plt.savefig(save_path)

#plt.show()


plt.plot(results.history['loss'])#train_loss
plt.title('train_loss')
plt.figure()
plt.savefig(save_path)

#plt.show()
plt.plot(results.history['val_loss'])
plt.title('val_loss')
plt.figure()
plt.savefig(save_path)

#plt.show()
plt.plot(results.history['accuracy'])
plt.title('train_accuracy')
plt.figure()
plt.savefig(save_path)

#plt.show()
plt.plot(results.history['val_accuracy'])
plt.title('val_accuracy')
plt.figure()
plt.savefig(save_path)

#plt.show()

#comparisons
plt.plot(results.history['loss'])#train_loss
plt.plot(results.history['val_loss'])
plt.title('train_loss & val_loss')
plt.figure()
plt.savefig(save_path)

#plt.show()

plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('train_accuracy & val_accuracy')
plt.figure()
plt.savefig(save_path)

#plt.show()


####################################

idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
##imshow(X_train[ix])
plt.plot(X_train[ix])
plt.figure()
plt.savefig(save_path)

#plt.show()
##imshow(np.squeeze(Y_train[ix]))
plt.plot(np.squeeze(Y_train[ix]))
plt.figure()
plt.savefig(save_path)

#plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.figure()
plt.savefig(save_path)

#plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
##imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.plot(X_train[int(X_train.shape[0]*0.9):][ix])
plt.figure()
plt.savefig(save_path)

# imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.plot(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.figure()
plt.savefig(save_path)
#plt.show()

plt.figure()
# imshow(np.squeeze(preds_val[ix]))
plt.plot(np.squeeze(preds_val[ix]))
plt.savefig(save_path)
#plt.show()

plt.figure()
ix = random.randint(0, len(preds_test_t))
##imshow(X_test[ix])
plt.plot(X_test[ix])
plt.savefig(save_path)
#plt.show()

plt.figure()
##imshow(np.squeeze(preds_test[ix]))
plt.plot(np.squeeze(preds_test[ix]))
plt.savefig(save_path)

#plt.show()


plt.figure()
ix = random.randint(0, len(preds_test_t))
imshow(X_test[ix])
# plt.plot(X_test[ix])

plt.savefig(save_path)

#plt.show()
plt.figure()
A = np.squeeze(preds_test[ix])
mid_point = int (((np.max(A) + np.min(A)) / 2 ) * 255 )
img_t1 = cv2.threshold(A*255.0, mid_point, 255, cv2.THRESH_BINARY_INV)
##imshow(img_t1[1].astype('uint8'),cmap='binary')
plt.plot(img_t1[1].astype('uint8'),cmap='binary')

plt.savefig(save_path)

#plt.show()



ix = random.randint(0, len(preds_test_t))
##imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.plot(X_train[int(X_train.shape[0]*0.9):][ix])
plt.figure()
plt.savefig(save_path)

#plt.show()
A = np.squeeze(preds_test[ix-1,:,:,:])
mid_point = int (((np.max(A) + np.min(A)) / 2 ) * 255 )

img_t1 = cv2.threshold(A*255.0, mid_point, 255, cv2.THRESH_BINARY_INV)

##imshow(img_t1[1].astype('uint8'),cmap='binary')
plt.plot(img_t1[1].astype('uint8'),cmap='binary')
plt.figure()
plt.savefig(save_path)

#plt.show()


# ####!tensorboard --logdir=logs/ --host localhost --port 8088

