import torch
from mytools import *
import torch.nn as nn
from tdnn import TDNN as TDNNLayer

class TDNNv1(nn.Module):
    '''
    TDNN Model from Paper, consisting of the following layers:
    - tdnn 1: 16 in channels, 8 out channels, 15 samples, window of 3
    - sigmoid after tdnn
    - tdnn 2: 8 in channels, 3 out channels, 13 samples, window of 5
    - sigmoid after tdnn
    - flatten: 9 frequencies, 3 out channels, flattens to (27, ) array
    - linear: 27 inputs, 4 outputs
    '''

    def __init__(self):
        super(TDNNv2, self).__init__()

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
    

model_params_path = 'model_params/model_params_015' 
model_params = torch.load(model_params_path)
model = model_params['model']
mean = model_params['mean']
std = model_params['std']

model.eval()
model = model.float()

single_sample = np.zeros((16, 15))
single_sample[10][9] = 1

data_tensor = torch.tensor([single_sample])

tdnnweight1 = None
tdnnbias1 = None
for name, param in model.state_dict().items():
    # name: str
    # param: Tensor
    arr = param.numpy()
    # print(name, arr.shape)
    if name == 'tdnn1.temporal_conv.weight':
        tdnnweight1 = arr
    if name == 'tdnn1.temporal_conv.bias':
        tdnnbias1 = arr

std = torch.std(data_tensor, 0)
mean = torch.mean(data_tensor, 0)
normed_data = (data_tensor - mean)/std # normalize 


print('input shape', data_tensor.shape)
print(data_tensor)
tdnn1 = torch.nn.Conv1d(16, 8,
                kernel_size=3,
                dilation=1,
                padding=0,
                bias=True   # will be set to False for semi-orthogonal TDNNF convolutions
        )
tdnn1.weight.data = torch.tensor(tdnnweight1)
tdnn1.bias.data = torch.tensor(tdnnbias1)
o1 = tdnn1(data_tensor.float())
print('after first conv', o1.shape)
print(o1)
o2 = TDNNLayer(8, 3, [-2,0,2])(o1)
print('after first conv', o2.shape)

'''
TESTING PYTORCH LINEAR LAYER
'''
sample1 = torch.tensor([[0, 1, 0]], dtype=float)
# print(sample1)
dense = nn.Linear(3, 2)
dense.weight.data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=float)
dense.bias.data = torch.tensor([0, 0], dtype=float)
print('sample 1 shape for linear layer', sample1.shape)
# print('weight bias shapes', dense.weight.data.shape, dense.bias.data.shape)
out1 = dense(sample1)
print('\n pytorch linear output \n',out1)

'''
TESTING TENSORFLOW LINEAR LAYER
'''
from tensorflow import keras as tf_ks
import numpy as np

tf_dense = tf_ks.layers.Dense(2)
weight1 = np.array([[1, 4], [2, 5], [3, 6]], dtype=float)
bias1 = np.array([0, 0], dtype=float)
sample1 = np.array([[0, 1, 0]], dtype=float)
dummy = tf_dense(sample1)
tf_dense.set_weights([weight1, bias1])
print('\n sample 1 shape for linear layer', sample1.shape)
print('\n tensorflow linear output \n', tf_dense(sample1))

'''
TESTING PYTORCH CONV1D LAYER
'''

sample2 = [i for i in range(15)]
sample2 = np.reshape(sample2, [5, 3])
print('\n sample 2 shape for conv1d', sample2.shape)

torch_conv1d = torch.nn.Conv1d(5, 2,
                kernel_size=1,
                dilation=1,
                padding=0,
                bias=True   # will be set to False for semi-orthogonal TDNNF convolutions
        )
weight2 = np.zeros([2, 5, 1], dtype=float)
weight2[0][3][0] = 1
bias2 = np.zeros([2], dtype=float)

torch_conv1d.weight.data = torch.tensor(weight2, dtype=float)
torch_conv1d.bias.data = torch.tensor(bias2, dtype=float)
o2 = torch_conv1d(torch.tensor(sample2, dtype=float))
print('\n pytorch conv1d output\n', o2)

'''
TESTING TF CONV1D LAYER
'''
tf_conv1d = tf_ks.layers.Conv1D(
    filters=2,
    kernel_size=1,
    dilation_rate=1,
    padding='valid'
)
sample2 = np.transpose(sample2)
sample2 = np.expand_dims(sample2, 0)
sample2 = np.array(sample2, dtype=float)

print('\n sample 2 shape for conv1d layer', sample2.shape)
dummy = tf_conv1d(sample2)
weight2 = np.transpose(weight2, [2, 1, 0])
tf_conv1d.set_weights([weight2, bias2])
out2 = tf_conv1d(sample2)
print('\n tensorflow conv1d output \n', out2)
