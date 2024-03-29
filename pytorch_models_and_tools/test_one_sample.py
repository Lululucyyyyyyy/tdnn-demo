import torch
from tdnn import TDNN as TDNNLayer
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mytools import *

class TDNNv1(nn.Module):
    def __init__(self):
        super(TDNNv1, self).__init__()

        self.tdnn1 = TDNNLayer(16, 8, [-1,0,1])
        self.sigmoid1 = nn.Sigmoid()
        self.tdnn2 = TDNNLayer(8, 3, [-2,0,2])
        self.sigmoid2 = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(27, 3)
        self.network = nn.Sequential(
            self.tdnn1,
            self.sigmoid1,
            self.tdnn2,
            self.sigmoid2,
            self.flatten,
            self.linear,
        )

    def forward(self, x):
        out = self.network(x)
        return out

logged = True
normed = False

# dat = []
# dat_lab = []
# fs = []
# load_file('ba00000-300.txt', 0, 'b', 'dataset9/', dat, dat_lab, fs)
# # print(np.array(dat).shape)
# data = dat[0]
spectrogram_file_path = "b00001-500.txt"
start = 50

with open(spectrogram_file_path) as g:
    mel_spectrogram = [[float(num) for num in line.split(',')][start:start + 15] for line in g]
g.close()

print('mel_spectrogram')
# print(mel_spectrogram)

data = torch.Tensor(mel_spectrogram)
print(data.shape)

# loading model params from file
model_params_path = 'model_params/model_params_023' 
model_params = torch.load(model_params_path)
model = model_params['model']
mean = model_params['mean']
std = model_params['std']
model.eval()

mean = torch.tensor(mean)
std = torch.tensor(std)

for name, param in model.state_dict().items():
    arr = param.numpy()
    # print(name, arr.shape)
    # print(arr)

# data processing with same parameters as model
data_tensor = torch.Tensor(data)
print('data tensor shape' ,data_tensor.shape)
print(data_tensor)
logged_data = data_tensor
if (not logged):
    logged_data = torch.log(data_tensor).max(torch.tensor(-25))
normed_data = logged_data
if (not normed):
  normed_data = (logged_data - mean)/std
inputs = normed_data
# print(inputs)

inputs = torch.tensor(inputs)
# for name, param in model.state_dict().items():
#     print(name)
#     print(param)

# print('mean')
# print(mean)
# print('std')
# print(std)

# print(inputs.shape)
curr_input = inputs.unsqueeze(0)
# print(curr_input)
outputs = model(curr_input)
_, predicted = torch.max(outputs.data, 1)

print('outputs', outputs)
print('predicted', predicted)