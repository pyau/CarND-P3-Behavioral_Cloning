import cv2
import csv
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Dropout
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import sklearn


samples = []
images = []
measurements = []
correction = 0.1
start_y=50
end_y=140
BATCH_SIZE=8
FLIP_IMAGE=True
USE_SIDE_IMAGES=True
NUM_EPOCH = 3
folder_name='set4'

img_height=end_y-start_y
range_end = 1
batch_size_multiplier = 1
if USE_SIDE_IMAGES == True:
    range_end = 3
    batch_size_multiplier *= 3
if FLIP_IMAGE == True:
    batch_size_multiplier *= 2

with open(folder_name+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # use center, left and right images24108
                for i in range(range_end):
                    name = folder_name+'/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    # trim image to only see section with road
                    trim_img = image[start_y:end_y, :, :]
                    angle = float(batch_sample[3])
                    if i == 1:
                        angle += correction
                    elif i == 2:
                        angle -= correction
                    # also use the flip of current image ang angle
                    images.append(trim_img)
                    angles.append(angle)
                    if FLIP_IMAGE == True:
                        images.append(np.fliplr(trim_img))
                        angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# linear regression
def linearRegression():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(img_height, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model

# lenet
def lenet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(img_height, 320, 3)))
    model.add(Conv2D(6, 5, activation="relu", padding='same', name='conv1'))
    model.add(MaxPooling2D(2, name='pool1'))
    model.add(Conv2D(6, 5, activation="relu", padding='same', name='conv2'))
    model.add(MaxPooling2D(2, name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(img_height, 320, 3)))
    model.add(Conv2D(24, 5, activation="relu", padding='same', name='conv1'))
    model.add(MaxPooling2D(2, name='pool1'))
    #model.add(Dropout(0.8))
    model.add(Conv2D(36, 5, activation="relu", padding='same', name='conv2'))
    model.add(MaxPooling2D(2, name='pool2'))
    #model.add(Dropout(0.8))
    model.add(Conv2D(48, 5, activation="relu", padding='same', name='conv3'))
    model.add(MaxPooling2D(2, name='pool3'))
    #model.add(Dropout(0.8))
    model.add(Conv2D(64, 3, activation="relu", padding='same', name='conv4'))
    model.add(Conv2D(64, 3, activation="relu", padding='same', name='conv5'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = nvidia()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(
    train_generator, \
    samples_per_epoch= len(train_samples)*batch_size_multiplier/BATCH_SIZE, \
    validation_data=validation_generator, \
    nb_val_samples=len(validation_samples)*batch_size_multiplier/BATCH_SIZE, \
    steps_per_epoch = len(train_samples)/BATCH_SIZE, \
    nb_epoch=NUM_EPOCH)

model.save('model.h5')

print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

import gc
gc.collect()

