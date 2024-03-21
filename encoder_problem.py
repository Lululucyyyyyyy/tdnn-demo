from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from mytools import *
import argparse

import tensorflow as tf
import numpy as np
import cv2

'''
Encoder Model
'''

# model = tf.keras.Sequential()

# model.add(tf.keras.layers.Dense(
#         units=2,
#         # input_shape = (5,),
#         activation='sigmoid'))

# model.add(tf.keras.layers.Dense(
#         units=5,
#         # input_shape=(2,),
#         activation='sigmoid'))

# n_classes = 5
# # n_samples = 100
# inputs = np.eye(n_classes)#[np.random.choice(n_classes, n_samples)]
# labels = inputs

# opt = tf.keras.optimizers.Adam(learning_rate=0.1)
# # loss = tf.keras.losses.CategoricalCrossentropy()
# model.compile(opt, loss = 'mse', metrics=['accuracy'])

# model.fit(inputs, labels,
#         epochs=1000)

# scores = model.evaluate(inputs, labels, verbose=0)
# out = model.predict(inputs)
# print('scores', scores)
# print('out', out)

'''
Simple Convolution
'''
from os import listdir
from os.path import isfile, join

num_files = 55

# cats = 1, dogs = 0
cats_dogs_data = []
for i in range(num_files):
    cat = cv2.imread('cats_dogs/PetImages/Cat/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE) 
    cat = cv2.resize(cat, (100, 100))
    cats_dogs_data.append(cat)

for i in range(num_files):
    dog = cv2.imread('cats_dogs/PetImages/Dog/' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE) 
    dog = cv2.resize(dog, (100, 100))
    cats_dogs_data.append(dog)

inputs = np.array(cats_dogs_data)#.reshape((num_files * 2, 100, 100, 1))
labels = np.hstack((np.ones(num_files), np.zeros(num_files))).reshape((num_files * 2, 1))
print(f'inputs.shape: {inputs.shape}, labels.shape: {labels.shape}')

test_ratio = 0.2
num_train = int(inputs.shape[0] * (1 - test_ratio))
indices = np.random.permutation(inputs.shape[0])
training_idx, test_idx = indices[:num_train], indices[num_train:]

train_inputs, train_labels = inputs[training_idx, :], labels[training_idx, :]
test_inputs, test_labels = inputs[test_idx, :], labels[test_idx, :]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(input_shape=(100, 100),
                                 filters=8,
                                 kernel_size=3,
                                 activation='sigmoid'))
model.add(tf.keras.layers.Conv1D(input_shape=(100, 100),
                                 filters=3,
                                 kernel_size=3,
                                 activation='sigmoid'))

# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(100, 100, 1)))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# # block 2
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# # block 3
# model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# model.add(tf.keras.layers.Conv1D(input_shape=[15, 16],
#                             filters = 8,
#                             kernel_size= 3,
#                             dilation_rate= 1, # 0 - (-1)
#                             padding= 'valid' # no padding
#                             ))

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_inputs, train_labels,
        epochs=50)

scores = model.evaluate(train_inputs, train_labels, verbose=0)
# out = model.predict(inputs)
print('scores', scores)
# print('out', out)

out = model.predict(test_inputs)
print('model predicts:', out)
print('true labels:', test_labels)