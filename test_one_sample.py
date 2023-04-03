from samples_from_web.sample1 import sample1
from samples_from_web.sample2 import sample2
from samples_from_web.sample3 import sample3
logged = True
normed = True

data = sample1

import torch
from tdnn import TDNN as TDNNLayer
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

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

# loading model params from file
model_params_path = 'model_params/model_params_011' 
model_params = torch.load(model_params_path)
model = model_params['model']
mean = model_params['mean']
std = model_params['std']
model.eval()

# data processing with same parameters as model
data_tensor = torch.Tensor(data)
logged_data = torch.transpose(data_tensor, 0, 1)
if (not logged):
		logged_data = torch.log(data_tensor).max(torch.tensor(-25))
normed_data = logged_data
if (not normed):
  normed_data = (logged_data - mean)/std
inputs = normed_data

output_b = []
output_d = []
output_g = []
output_null = []

timeline = [i for i in range(len(inputs))]

curr_input = inputs.reshape([1, 16, 15])
outputs = model(curr_input)
_, predicted = torch.max(outputs.data, 1)

print('outputs', outputs)
print('predicted', predicted)

# fig, ax = plt.subplots()
# ax.set_title('melSpectrogram of ' + file_name)

# line1, = ax.plot(output_b, color="skyblue", label="b")	
# line2, = ax.plot(output_d, color="mediumslateblue", label="d")	
# line3, = ax.plot(output_g, color="plum", label="g")	
# line4, = ax.plot(output_null, color="red", label="null")	
# plt.plot([time] * 2, [-10, 10], color="khaki")

# ax.legend([line1, line2, line3, line4], ["b", "d", "g", "null"])
# plt.show()

