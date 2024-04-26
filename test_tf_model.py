import tensorflow as tf
from mytools import *
import keras.backend as K

model = tf.keras.models.load_model('model3.keras')

# Show the model architecture
model.summary()

f = 'bah-230.txt'
# f = 'dataset9/g/ga00000-340.txt'
# f = 'guh00000-690.txt'
data = []

load_file(f, label=0, letter='', path_to_training_data=None, data=data)


data_normed = []
for dat in data:
    mean, std = np.mean(dat), np.var(data) ** 0.5
    assert(len(mean.shape) < 1 and len(std.shape) < 1)
    normed_dat = np.divide(np.subtract(dat, mean), std)# normalize 
    normed_dat = normed_dat.transpose()
    data_normed.append(normed_dat)
normed_data = np.array(data_normed)
data = normed_data

print(model.predict(data))

inputs = data
for layer in model.layers:
    inputs = layer(inputs)
    # print(inputs)
    # print(layer.get_weights()[1])