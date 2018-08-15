import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

train_fname = "/Users/neetipandey/Documents/Kaggle/Digit-Recognition/data/train.csv"
test_fname = "../test.csv"
output_fname = "../result.csv"


# skiprows=1 : skip reading and converting col labels
data = np.loadtxt(train_fname,  dtype=int, skiprows=1, delimiter=",")

# Define features X and labels y
X, y = data[:, 1:], data[:, 0]
# split the data into train & validate
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
print(X_train.shape)

#to work with 28x28 data in CNN, reshape the data from 1x784 to 28x28

#X_train = np.reshape(X_train, (-1, 28, 28))
#X_valid = np.reshape(X_valid, (-1, 28, 28))

X_train = X_train.reshape(-1, 28, 28, 1)
X_valid = X_valid.reshape(-1, 28, 28, 1)

print(X_train.shape)




#Nonrmalize values of pixels to have values between 0-1 instead of 0-255
# for faster convergence
X_train = X_train.astype(dtype="float32")/255
X_valid = X_valid.astype(dtype="float32")/255



#One-hot-encoding for values to fit the categorical classification
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)


print(y_train[0])

############ TRAIN THE MODEL ####################################

model = Sequential() #adding on one layer at a time

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization()) #helps train faster

model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)
# logloss loss function
# Adam, is faster than SGD
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])

# train once with a smaller learning rate to ensure convergence.
# then speed things up, only to reduce the learning rate by 10% every epoch
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=30, #20 (tried) for Kaggle kernel
                           verbose=2,  #1 for ETA, 0 for silent
                           validation_data=(X_valid[:400,:], y_valid[:400,:]), #For speed
                           callbacks=[annealer])






#
