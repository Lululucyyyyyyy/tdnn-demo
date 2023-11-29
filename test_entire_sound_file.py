'''
Displays a plot of the different predicted values across
an entire sound file
'''

import torch
from tdnn import TDNN as TDNNLayer
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

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
    
class TDNNv2(nn.Module):
    def __init__(self):
        super(TDNNv2, self).__init__()

        self.tdnn1 = TDNNLayer(16, 8, [-1,0,1])
        self.sigmoid1 = nn.Sigmoid()
        self.tdnn2 = TDNNLayer(8, 3, [-2,0,2])
        self.sigmoid2 = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(27, 3)
        self.sigmoid3 = nn.Sigmoid()
        self.network = nn.Sequential(
            self.tdnn1,
            self.sigmoid1,
            self.tdnn2,
            self.sigmoid2,
            self.flatten,
            self.linear,
            # self.sigmoid3,
        )

    def forward(self, x):
        out = self.network(x)
        return out
    
def parse_arguments():
    # Command-line flags are defined here.
    parser = ArgumentParser()
    parser.add_argument('--file', dest='file', type=str,
                        default="", help="File of Spectrogram")
    parser.add_argument('--model', dest='model', type=str,
                     default="model_params/model_params_015", help="Model to use")
    parser.add_argument('--verbose', dest="verbose", type=bool,
                        default=False, help="print useless stuffs")
    return parser.parse_args()

args = parse_arguments()

# loading model params from file
model_params_path = args.model
model_params = torch.load(model_params_path)
model = model_params['model']
mean = model_params['mean']
std = model_params['std']

model.eval()

# testing
# spectrogram_file_path = 'dataset9/b/ba00000-300.txt'
spectrogram_file_path = args.file
fs = spectrogram_file_path.split("-") # split file name to get starting frame
file_name = spectrogram_file_path.split('.')[0]
time = int(file_name.split('-')[1]) / 10

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
# logged_data = 10*torch.log10(data_tensor + 10**-25)
normed_data = (data_tensor - mean)/std
inputs = normed_data

# print('data')
# print(data_tensor)
# print('inputs')
# print(inputs)

output_b = []
output_d = []
output_g = []
output_null = []

timeline = [i for i in range(len(inputs))]

for i, curr_input in enumerate(inputs):
    curr_input = curr_input.reshape([1, 16, 15])
    outputs = model(curr_input)
    _, predicted = torch.max(outputs.data, 1)
    output_b.append(outputs[0][0].item())
    output_d.append(outputs[0][1].item())
    output_g.append(outputs[0][2].item())
    # output_null.append(outputs[0][3].item())
    if i == time:
        print('model predicted at true timeframe', predicted)
        # print('melspectrogram:')
        # print(curr_input)

fig, ax = plt.subplots()
ax.set_title('melSpectrogram of ' + file_name)

line1, = ax.plot(output_b, color="skyblue", label="b")	
line2, = ax.plot(output_d, color="mediumslateblue", label="d")	
line3, = ax.plot(output_g, color="plum", label="g")	
line4, = ax.plot(output_null, color="red", label="null")	
plt.plot([time] * 2, [-10, 10], color="khaki")

ax.legend([line1, line2, line3, line4], ["b", "d", "g", "null"])
plt.show()
