import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import keras
import os, sys
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import numpy as np


# Define train data
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
train_images_dir = os.path.join('./', 'cars_train')
train_labels_mat = scipy.io.loadmat('cars_train_annos.mat')
train_labels_data = [[row.flat[0] for row in line] for line in train_labels_mat['annotations'][0]]
df_train = pd.DataFrame(train_labels_data, columns=columns)
print("\n================================\nTrain Data\n================================\n", df_train)

# Define test data
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
test_images_dir = os.path.join('./', 'cars_test')
test_labels_mat = scipy.io.loadmat('cars_test_annos.mat')
test_labels_data = [[row.flat[0] for row in line] for line in test_labels_mat['annotations'][0]]
df_test = pd.DataFrame(test_labels_data, columns=columns) 
print("\n================================\nTest Data\n================================\n", df_test)

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width, img_height = 256, 256
nb_train_samples = 6114
nb_validation_samples = 2000
batch_size = 42
epochs = 100

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))


# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(196, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_image_generator = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=45,
                            width_shift_range=.15,
                            height_shift_range=.15,
                            horizontal_flip=True,
                            zoom_range=0.5
                        )

test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_images_dir,
                                                           target_size=(img_height, img_width),
                                                           shuffle=True,
                                                           class_mode='categorical')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=test_images_dir,
                                                             target_size=(img_height, img_width),
                                                             class_mode='categorical')




# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

logger = keras.callbacks.TensorBoard(log_dir="logs", write_graph=True, histogram_freq=5)

# Train the model 
model_final.fit_generator(
train_data_gen,
samples_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = test_data_gen,
nb_val_samples = nb_validation_samples,
callbacks = [checkpoint, early, logger])
