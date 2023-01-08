import os
import argparse
import numpy as np
from numba import cuda
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imread, imshow

parser = argparse.ArgumentParser()
parser.add_argument('--run_path', help='Path to the run folder', type=str)
parser.add_argument('--data_path', help='Path to the data folder', type=str)
parser.add_argument('--batch', help='batch size', type=int)
parser.add_argument('--shape', help='img_shape', type=int)
parser.add_argument('--learning', help='learning rate', default= 0.001, type=float)
parsed_args = parser.parse_args()

run_path = parsed_args.run_path
data_path = parsed_args.data_path
batch = parsed_args.batch
shape = parsed_args.shape
learning = parsed_args.learning

try:
    os.mkdir(run_path)
except:
    print("")

IMG_WIDTH = shape
IMG_HEIGHT = shape
IMG_CHANNELS = 3

load = os.path.exists(os.path.join(data_path, "Xtrain_"+str(shape)+".npy"))

if load == False:

    TRAIN_images_PATH = os.path.join(data_path, "train/img")
    TRAIN_masks_PATH =  os.path.join(data_path, "train/mask")
    VAL_images_PATH =   os.path.join(data_path, "val/img")
    VAL_masks_PATH =    os.path.join(data_path, "val/mask")

    train_images_list = sorted(os.listdir(TRAIN_images_PATH))
    train_masks_list = sorted(os.listdir(TRAIN_masks_PATH))
    val_images_list = sorted(os.listdir(VAL_images_PATH))
    val_masks_list = sorted(os.listdir(VAL_masks_PATH))


    # train images and masks
    X_train = np.zeros((len(train_images_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    print('Loading train images') 
    for n, id_ in enumerate(train_images_list):
        path = os.path.join(TRAIN_images_PATH, id_)
        img = imread(path)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
    Y_train = np.zeros((len(train_masks_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
    print('Loading masks images') 
    for n, id_ in enumerate(train_masks_list):
        path = os.path.join(TRAIN_masks_PATH, id_)
        mask = imread(path)[:,:,:1]
        mask = (resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True))
        Y_train[n] = mask
    np.save(os.path.join(data_path, "Xtrain_"+str(shape)),X_train)
    np.save(os.path.join(data_path, "Ytrain_")+str(shape),Y_train)

    # val images and masks
    X_val = np.zeros((len(val_images_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    print('Loading val images') 
    for n, id_ in enumerate(val_images_list):
        path = os.path.join(VAL_images_PATH, id_)
        img = imread(path)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_val[n] = img
    Y_val = np.zeros((len(val_masks_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
    print('Loading masks images') 
    for n, id_ in enumerate(val_masks_list):
        path = os.path.join(VAL_masks_PATH, id_)
        mask = imread(path)[:,:,:1]
        mask = (resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True))
        Y_val[n] = mask
    np.save(os.path.join(data_path, "Xval_"+str(shape)),X_val)
    np.save(os.path.join(data_path, "Yval_"+str(shape)),Y_val)

if load == True:
    print('Loading numpys') 
    X_train=np.load(os.path.join(data_path, "Xtrain_"+str(shape)+".npy"),allow_pickle=True)
    Y_train=np.load(os.path.join(data_path, "Ytrain_"+str(shape)+".npy"),allow_pickle=True)
    X_val=np.load(os.path.join(data_path, "Xval_"+str(shape)+".npy"),allow_pickle=True)
    Y_val=np.load(os.path.join(data_path, "Yval_"+str(shape)+".npy"),allow_pickle=True)

device = cuda.get_current_device()
device.reset()

#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

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
#utputs = tf.keras.layers.Conv2D(1,(1,1), activation='softmax')(c9) # TODO test??
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=learning), loss='binary_crossentropy', metrics=['accuracy', 'mse', 'mae', 'mape','Precision','Recall'])

model.summary()

#save checkpoints
checkpointer = tf.keras.callbacks.ModelCheckpoint('ckpt.h5', verbose=1, save_best_only=True) 

callbacks = [tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss'), tf.keras.callbacks.TensorBoard(log_dir=run_path)]

print("training")
results = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=batch, epochs=300, callbacks=callbacks)

tf.keras.models.save_model(model,os.path.join(run_path, "model.h5"))




