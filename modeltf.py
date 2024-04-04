from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from mytools import *
import argparse

import tensorflow as tf
import numpy as np

'''
To convert the model, run:
tensorflowjs_converter --input_format=keras-saved-model model0.keras /web_model/tfjs_model
tensorflowjs_converter --input_format=keras --output_format tfjs_layers_model  model2.h5 web_model/tfjs_model
'''

def load_and_process_data():
    '''
    Training Data set: 
    - labels 0 = b, 1 = d, 2 = g
    '''
    data = []
    data_labels = [] 
    files = []

    load_dataset("dataset9/", data, data_labels, files, 9, 0)
    load_dataset("dataset9/other_data/", data, data_labels, files, 9, 0)

    assert(len(data) == len(data_labels) and len(data) == len(files))

    '''
    Data Processing
    '''
    
    test_length = int(len(data) * 0.1)
    train_length = len(data) - test_length

    # normalize
    # mean, var = np.mean(data_tensor, axis=0), np.var(data_tensor, axis=0)
    # std = var ** 0.5
    # normed_data = np.divide(np.subtract(data_tensor, mean), std)# normalize 
    # normed_data = normed_data.transpose(0, 2, 1)
    data_normed = []
    for dat in data:
        mean, std = np.mean(dat), np.var(data) ** 0.5
        assert(len(mean.shape) < 1 and len(std.shape) < 1)
        normed_dat = np.divide(np.subtract(dat, mean), std)# normalize 
        normed_dat = normed_dat.transpose()
        data_normed.append(normed_dat)
    normed_data = np.array(data_normed)

    # process labels and data loader
    labels_tensor = np.array(data_labels, dtype=int)
    labels_tensor = np.eye(3)[labels_tensor]
    # whole_dataset = tf.data.Dataset.from_tensor_slices((normed_data, labels_tensor))


    # train_dataset, test_dataset = torch.utils.data.random_split(whole_dataset, [train_length, test_length])

    print("dataset done loading; loaded total of ", len(data), "samples")
    print(sum([0 if x != 0 else 1 for x in data_labels]), "b's, label:0")
    print(sum([0 if x != 1 else 1 for x in data_labels]), "d's, label:1")
    print(sum([0 if x != 2 else 1 for x in data_labels]), "g's, label:2")
    print(sum([0 if x != 3 else 1 for x in data_labels]), "null vals, label:3")
    print('========== hyperparameters ============')
    print("number of training examples:", train_length)

    return normed_data, labels_tensor, mean, std
    # return train_length, test_length, mean, std, files_dict, kfold, train_dataset, test_dataset

# num_folds = 1

# kfold = KFold(n_splits=num_folds, shuffle=True)

inputs, targets, mean, std = load_and_process_data()
# K-fold Cross Validation model evaluation
# fold_no = 1
num_epochs = 4000
learning_rate = 0.01
momentum = 0.1
batch_size = 50
verbosity = False
acc_per_fold = []
loss_per_fold = []
test_ratio = 0.1


# for train, test in kfold.split(inputs, targets):
num_train = int(inputs.shape[0] * (1 - test_ratio))
indices = np.random.permutation(inputs.shape[0])
training_idx, test_idx = indices[:num_train], indices[num_train:]

train_inputs, train_labels = inputs[training_idx, :], targets[training_idx, :]
test_inputs, test_labels = inputs[test_idx, :], targets[test_idx, :]

# Define the model architecture
model = tf.keras.Sequential()

# Adding tdnn1
# context1 = [-1, 0, 1]
model.add(tf.keras.layers.Conv1D(input_shape=[15, 16],
                            filters = 8,
                            kernel_size= 3,
                            dilation_rate= 1, # 0 - (-1)
                            padding= 'valid', # no padding
                            activation='sigmoid',
                            #kernel_initializer=tf.keras.initializers.glorot_normal()
                            ))

# Adding tdnn2
# context2 = [-2, 0, 2]
model.add(tf.keras.layers.Conv1D(input_shape=[13, 8],
                            filters = 3,
                            kernel_size= 5,
                            dilation_rate= 2, # 0 - (-1)
                            padding= 'valid', # no padding
                            activation='sigmoid',
                            #kernel_initializer=tf.keras.initializers.glorot_normal()
                            ))

# Adding Flatten
model.add(tf.keras.layers.Flatten()) #data_format='channels_first'

#// Adding Linear
model.add(tf.keras.layers.Dense(
    units=3,
    activation=None,
    use_bias=True))

# Compile the model
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
# opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
# loss = tf.keras.losses.CategoricalCrossentropy()
loss = tf.keras.losses.MeanSquaredError()
model.compile(opt, loss, metrics=['accuracy'])


# Generate a print
# print('------------------------------------------------------------------------')
# print(f'Training for fold {fold_no} ...')

# Fit data to model
history = model.fit(train_inputs, train_labels,
                    validation_split=0.1,
                batch_size=batch_size,
                epochs=num_epochs)

predictions = model.predict(test_inputs)

model.save('model2.keras')
# tf.contrib.saved_model.save_keras_model('model1.keras') #outdated
model.save('model2.h5')

# save weights
model.save_weights('weights/my_weights')

# np.savetxt('model2_mean.txt', mean, fmt='%.18e', delimiter=', ', newline='],\n[', header='[', footer=']', comments='', encoding=None)
# np.savetxt('model2_std.txt', std, fmt='%.18e', delimiter=', ', newline='],\n[', header='[', footer=']', comments='', encoding=None)

# Generate generalization metrics
scores = model.evaluate(train_inputs, train_labels, verbose=0)
print(f'Train performance: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
acc_per_fold.append(scores[1] * 100)
loss_per_fold.append(scores[0])

scores = model.evaluate(test_inputs, test_labels, verbose=0)
print(f'Test performance: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
acc_per_fold.append(scores[1] * 100)
loss_per_fold.append(scores[0])

# Increase fold number
# fold_no = fold_no + 1

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0, 1)
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

train_preds = model.predict(train_inputs)
confusion = tf.math.confusion_matrix(labels=np.argmax(train_labels, axis=1), predictions=np.argmax(train_preds, axis=1))
print()
print('the confusion matrix of training dataset')
print(confusion)

confusion = tf.math.confusion_matrix(labels=np.argmax(test_labels, axis=1), predictions=np.argmax(predictions, axis=1))
print()
print('the confusion matrix of test set')
print(confusion)