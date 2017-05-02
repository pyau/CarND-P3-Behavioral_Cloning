import cv2
import csv
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Dropout
import matplotlib.pyplot as plt

images = []
measurements = []
correction = 0.2

def load_data(set_name, load_center_only):
    lines = []
    with open(set_name+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    if load_center_only == True:
        # only load center images
        for line in lines:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = set_name+'/IMG/' + filename
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
    else:
        # load all 3 sides
        for line in lines:
            for i in range(3):
                source_path = line[i]
                filename = source_path.split('/')[-1]
                current_path = set_name+'/IMG/' + filename
                image = cv2.imread(current_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                measurement = float(line[3])
                if i == 1:
                    measurement += correction
                elif i == 2:
                    measurement -= correction
                measurements.append(measurement)



load_data('set1', False)
load_data('set2', False)
#load_data('set3', False)
#load_data('set4', False)
load_data('set5', False)
load_data('set_recovery', False)
#load_data('set_reverse', False)
load_data('set_curve', False)
# ------------------------------------------------
# augment flip
# ------------------------------------------------
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))   #(cv2.flip(image,1))
    augmented_measurements.append(-measurement)     #(measurement*-1.0)

# ------------------------------------------------
# training
# ------------------------------------------------
X_train = np.array(augmented_images)  #(images)
y_train = np.array(augmented_measurements)    #(measurements)


# linear regression
def linearRegression():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model

# lenet
def lenet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # cropping the top and bottom useless rows
    model.add(Cropping2D(cropping=((50,20),(0,0))))
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
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # cropping the top and bottom useless rows
    model.add(Cropping2D(cropping=((50,20),(0,0))))
    model.add(Conv2D(24, 5, activation="relu", padding='same', name='conv1'))
    model.add(MaxPooling2D(2, name='pool1'))
    model.add(Conv2D(36, 5, activation="relu", padding='same', name='conv2'))
    model.add(MaxPooling2D(2, name='pool2'))
    model.add(Conv2D(48, 5, activation="relu", padding='same', name='conv3'))
    model.add(MaxPooling2D(2, name='pool3'))
    #model.add(Dropout(0.75))
    model.add(Conv2D(64, 3, activation="relu", padding='same', name='conv4'))
    model.add(Conv2D(64, 3, activation="relu", padding='same', name='conv5'))
    model.add(Flatten(name='flatten'))
    model.add(Dropout(0.75))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


#model = lenet()
model = nvidia()

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

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

