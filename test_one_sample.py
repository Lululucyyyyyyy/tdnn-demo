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
normed = True

# dat = []
# dat_lab = []
# fs = []
# load_file('ba00000-300.txt', 0, 'b', 'dataset9/', dat, dat_lab, fs)
# # print(np.array(dat).shape)
# data = dat[0]

data =   [[-1.185027 , -1.4773506, -1.1079551, -1.259577 , -3.0923717, -3.5491922, -3.5945659, -3.3703151, -4.0364094, -5.0473328, -4.1083226, -3.3133235, -1.6250907, -0.8243138, -1.2552584], [-0.3829722, -1.1696534, -1.9007236, -3.195147 , -3.0130465, -3.9543548, -2.7006245, -2.6400118, -2.1601365, -3.2260773, -3.1481662, -1.8290879, -2.7632015, -2.5523379, -2.380501 ], [-0.8824555, -2.229085 , -5.5887327, -8.4832449, -7.8281484, -7.8623023, -7.0550351, -6.4117646, -5.045156 , -4.9955392, -5.5953588, -4.8741121, -4.1080589, -4.0999527, -2.1485562], [-1.0537763, -2.9274638, -6.1471367, -6.8126807, -6.3444691, -6.9366393, -6.3349247, -6.0495005, -5.6742811, -4.9254656, -5.9363194, -5.9972091, -5.2051868, -4.5088525, -2.5542259], [-1.272859 , -2.9639716, -6.1076794, -5.9064946, -5.1231017, -6.3645363, -6.0438976, -5.362998 , -3.6013706, -3.7824259, -3.7636111, -3.7112634, -3.4717066, -3.0715187, -0.6877223], [-1.6395373, -2.5811408, -4.161252 , -4.6124835, -4.7467585, -4.5139647, -4.6010485, -4.6494799, -4.0270734, -3.9491162, -4.2543898, -4.0365419, -3.6716774, -3.2936022, -1.5271825], [-1.2617574, -2.1399717, -3.123899 , -3.6319199, -3.8732891, -3.7312982, -3.8086138, -4.0195527, -3.6126618, -3.7403276, -3.5707519, -3.714479 , -3.4144783, -2.8156998, -0.4784096], [-1.2985272, -1.9373944, -2.7923512, -3.0533133, -3.555032 , -3.7637286, -3.4654024, -3.3315716, -3.0518506, -3.2380819, -3.5909209, -3.420857 , -3.1178017, -2.4578688, -0.5867903], [-1.3930099, -2.2408605, -3.0274343, -3.5119419, -3.4973052, -3.5468161, -3.5487003, -3.5839086, -3.4690566, -3.3823931, -3.314728 , -3.173316 , -3.0368755, -2.8148897, -1.0390899], [-1.5019825, -2.4184399, -3.6327767, -4.0550723, -3.6987505, -3.6453338, -3.490659 , -3.5292313, -3.4800937, -3.3505943, -3.0003276, -2.843709 , -2.7354181, -2.3072701, -1.4416147], [-1.5698187, -2.5478916, -3.7163341, -4.0254946, -3.7785187, -3.8626032, -3.8185229, -3.7475405, -3.4518704, -3.3476038, -3.0485291, -3.0321879, -2.5492349, -1.9966112, 0.0852742 ], [-1.4153187, -1.8708049, -3.1481535, -2.871711 , -2.3927627, -2.4356608, -2.349133 , -2.3429873, -2.5326922, -2.7484829, -2.5164025, -2.8099318, -2.720427 , -1.4173386, 0.4033844 ], [-2.4432762, -1.8687447, -2.592967 , -2.714267 , -2.6833017, -3.1241033, -2.3226204, -2.1644664, -2.0831859, -2.3349648, -3.1468985, -2.3189955, -2.2334473, -2.1020124, 0.4617157 ], [-1.088861 , -1.6148033, -2.3784976, -2.7869458, -3.338722 , -3.5277436, -2.4906228, -2.370465 , -2.2353969, -2.5606 , -2.8457477, -2.2798028, -2.3225248, -1.8486111, 0.4597093 ], [-0.8065996, -1.6840652, -3.1229541, -3.3670626, -3.1144576, -2.8427849, -2.768883 , -3.0857978, -2.9118662, -2.8989511, -2.9811757, -2.6578512, -2.3186572, -1.5714225, 0.5105933 ], [-0.9002268, -2.0163224, -3.8239734, -3.7245126, -3.1218433, -2.9124353, -2.5748062, -2.6963351, -3.8935504, -2.7764304, -2.6759527, -2.3322322, -2.1922138, -1.5262085, 0.4714421 ]]
data = torch.Tensor(data)

# loading model params from file
model_params_path = 'model_params/model_params_023' 
model_params = torch.load(model_params_path)
model = model_params['model']
mean = model_params['mean']
std = model_params['std']
model.eval()

for name, param in model.state_dict().items():
    arr = param.numpy()
    print(name, arr.shape)
    print(arr)

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