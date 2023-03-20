# from pytorch2keras.converter import pytorch_to_keras
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare
# import torch
# from tdnn import TDNN as TDNNLayer
# import torch.nn as nn
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
model_params_path = 'model_params/model_params_009' 
model_params = torch.load(model_params_path)
model = model_params['model']
mean = model_params['mean']
std = model_params['std']
# state_dict = model_params['state_dict']

# for name, param in model.state_dict().items():
#     # name: str
#     # param: Tensor
#     print(name)
#     print(param)
#     print()

trained_model = model
trained_model.load_state_dict(model.state_dict())
dummy_input = Variable(torch.randn(1, 16, 15))
print(dummy_input.shape)
torch.onnx.export(trained_model, dummy_input, "model1.onnx")

model = onnx.load('model1.onnx')
tf_rep = prepare(model) 

spectrogram_file_path = 'dataset5/spectrograms/d/dance00001-430.txt'
fs = spectrogram_file_path.split("-") # split file name to get starting frame
file_name = spectrogram_file_path.split("/")[3].split('.')[0]
start = int(file_name.split('-')[1]) // 10
sample = None

with open(spectrogram_file_path, 'r') as g:
    mel_spectrogram = [[float(num) for num in line.split(',')][start:start + 15] for line in g]
    sample = [mel_spectrogram]
g.close()
label = 1

output = tf_rep.run(np.asarray(sample, dtype=np.float32))
print('Result ', np.argmax(output))
