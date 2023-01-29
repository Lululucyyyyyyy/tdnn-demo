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
model_params_path = 'model_params/model_params_001' 
model_params = torch.load(model_params_path)
model = model_params['model']
mean = model_params['mean']
std = model_params['std']
# model = TDNNv1()
# optimizer = optim.SGD(*args, **kwargs)

# params = torch.load(model_params_path)
# model.load_state_dict(params['model_state_dict'])
# optimizer.load_state_dict(params['optimizer_state_dict'])
model.eval()

# testing
spectrogram_file_path = 'dataset5/spectrograms/b/ball00001-240.txt'
fs = spectrogram_file_path.split("-") # split file name to get starting frame
start = int(fs[1].split('.')[0])//10
with open(spectrogram_file_path) as g:
    mel_spectrogram = [[float(num) for num in line.split(',')] for line in g]
g.close()

data = []
for x in range(0, len(mel_spectrogram[0]) - 15 + 1):
	curr_ms = [line[x:x+15] for line in mel_spectrogram]
	data.append(curr_ms)

# data processing with same parameters as model
data_tensor = torch.Tensor(data)
logged_data = torch.log(data_tensor).max(torch.tensor(-25)) 
std = torch.std(logged_data, 0)
mean = torch.mean(logged_data, 0)
normed_data = (logged_data - mean)/std
inputs = normed_data

output_b = []
output_d = []
output_g = []

timeline = [i for i in range(len(inputs))]

for i, curr_input in enumerate(inputs):
	curr_input = curr_input.reshape([1, 16, 15])
	outputs = model(curr_input)
	_, predicted = torch.max(outputs.data, 1)
	output_b.append(outputs[0][0].item())
	output_d.append(outputs[0][1].item())
	output_g.append(outputs[0][2].item())

file_name = spectrogram_file_path.split("/")[3].split('.')[0]
time = int(file_name.split('-')[1]) / 10

fig, ax = plt.subplots()
ax.set_title('melSpectrogram of ' + file_name)

line1, = ax.plot(output_b, color="skyblue", label="b")	
line2, = ax.plot(output_d, color="mediumslateblue", label="d")	
line3, = ax.plot(output_g, color="plum", label="g")	
plt.plot([time] * 2, [-10, 10], color="khaki")

ax.legend([line1, line2, line3], ["b", "d", "g"])
plt.show()
