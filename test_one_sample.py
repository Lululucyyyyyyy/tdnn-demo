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

dat = []
dat_lab = []
fs = []
load_file('ba00000-300.txt', 0, 'b', 'dataset9/', dat, dat_lab, fs)
# print(np.array(dat).shape)
data = dat[0]

# loading model params from file
model_params_path = 'model_params/model_params_015' 
model_params = torch.load(model_params_path)
model = model_params['model']
mean = model_params['mean']
std = model_params['std']
model.eval()

# data processing with same parameters as model
data_tensor = torch.Tensor(data)
print(data_tensor.shape)
logged_data = data_tensor
if (not logged):
    logged_data = torch.log(data_tensor).max(torch.tensor(-25))
normed_data = logged_data
if (not normed):
  normed_data = (logged_data - mean)/std
inputs = normed_data
# print(inputs)
# for name, param in model.state_dict().items():
#     print(name)
#     print(param)

# print('mean')
# print(mean)
# print('std')
# print(std)

print(inputs.shape)
curr_input = inputs.reshape([1, 16, 15])
outputs = model(curr_input)
_, predicted = torch.max(outputs.data, 1)

print('outputs', outputs)
print('predicted', predicted)