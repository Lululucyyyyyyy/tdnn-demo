# from pytorch2keras.converter import pytorch_to_keras
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
# import onnx
# from onnx_tf.backend import prepare
# import torch
from tdnn import TDNN as TDNNLayer
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
model_params_path = 'model_params/model_params_013' 
model_params = torch.load(model_params_path)
model = model_params['model']
mean = model_params['mean']
std = model_params['std']
# state_dict = model_params['state_dict']

for name, param in model.state_dict().items():
    # name: str
    # param: Tensor
    with open("model_params/013/"+name+".txt", "w+") as f:
        arr = param.numpy()
        print(name, arr.shape)
        if len(arr.shape) > 2:
          arr = arr.reshape(arr.shape[0], -1)
          # to load back, use
          # load_original_arr = loaded_arr.reshape(
          # loaded_arr.shape[0], loaded_arr.shape[1] // arr.shape[2], arr.shape[2])
        np.savetxt("model_params/013/"+name+".txt", arr, delimiter=',')
    f.close()

with open("model_params/013/mean.txt", "w+") as f:
  arr = mean.numpy()
  if len(arr.shape) > 2:
    arr = arr.reshape(arr.shape[0], -1)
    # to load back, use
    # load_original_arr = loaded_arr.reshape(
    # loaded_arr.shape[0], loaded_arr.shape[1] // arr.shape[2], arr.shape[2])
  np.savetxt("model_params/013/mean.txt", arr, delimiter=',')
  f.close()

with open("model_params/013/std.txt", "w+") as f:
  arr = std.numpy()
  if len(arr.shape) > 2:
    arr = arr.reshape(arr.shape[0], -1)
    # to load back, use
    # load_original_arr = loaded_arr.reshape(
    # loaded_arr.shape[0], loaded_arr.shape[1] // arr.shape[2], arr.shape[2])
  np.savetxt("model_params/013/std.txt", arr, delimiter=',')
  f.close()
