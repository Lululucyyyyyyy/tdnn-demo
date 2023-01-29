from tdnn import TDNN as TDNNLayer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import math, random
import os, re
import torch.optim as optim
import matplotlib.pyplot as plt

class TDNNv1(nn.Module):
    '''
    TDNN Model from Paper, consisting of the following layers:
    - tdnn 1: 16 in channels, 8 out channels, 15 samples, window of 3
    - sigmoid after tdnn
    - tdnn 2: 8 in channels, 3 out channels, 13 samples, window of 5
    - sigmoid after tdnn
    - flatten: 9 frequencies, 3 channels, flattens to (27, ) array
    '''

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

'''
Training Data set: 
- labels 0 = b, 1 = d, 2 = g
'''
data = []
data_labels = [] 

def load_file(f, label, letter, path_to_training_data):
    if not f.endswith(".txt"):
        return
    fs = f.split("-") # split file name to get starting frame
    start = int(fs[1].split('.')[0])//10
    with open(path_to_training_data+letter+"/"+f, 'r') as g:
        mel_spectrogram = [[float(num) for num in line.split(',')][start:start + 15] for line in g]
        data.append(mel_spectrogram)
    g.close()
    data_labels.append(label)

def load_dataset(path_to_training_data):
    l_before = len(data)
    print("loading", path_to_training_data)
    for f in sorted(os.listdir(path_to_training_data + "b")):
        load_file(f, 0, "b", path_to_training_data)
    
    for f in sorted(os.listdir(path_to_training_data + "d")):
        load_file(f, 1, "d", path_to_training_data)
       
    for f in sorted(os.listdir(path_to_training_data + "g")):
        load_file(f, 2, "g", path_to_training_data)
    length = len(data) - l_before
    print("loaded", length, "samples")

load_dataset("dataset3/spectrograms/")
load_dataset("dataset4/spectrograms/")
load_dataset("dataset5/spectrograms/")
load_dataset("dataset6/spectrograms/")
load_dataset("dataset7/spectrograms/")

'''
Data Processing
'''
test_length = int(len(data) * 0.1)
data_tensor = torch.Tensor(data) # turn into torch tensor
logged_data = torch.log(data_tensor).max(torch.tensor(-25)) 
# # take the log, element-wise, cap at -25 to avoid -inf

# normalize
std = torch.std(logged_data, 0)
mean = torch.mean(logged_data, 0)
normed_data = (logged_data - mean)/std # normalize 

# process labels and data loader
labels_tensor = torch.tensor(data_labels, dtype=torch.long)
whole_dataset = TensorDataset(normed_data, labels_tensor)

train_dataset, test_dataset = torch.utils.data.random_split(whole_dataset, [len(data) - test_length, test_length])
batch_size = 50
num_batches = (len(data) - test_length) // batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # create dataloader object

print("dataset done loading; loaded total of ", len(data), "samples")
print("number of training examples:", len(data) - test_length)
print("batch size:", batch_size, ", number of batches:", num_batches)

'''
testset is 10%
'''
test_loader = DataLoader(test_dataset, batch_size=test_length, shuffle=True)


'''
Training
'''

tdnn = TDNNv1()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(tdnn.parameters(), lr=0.15, momentum=0.2)

# hyperparameters
num_epochs = 600
losses = []

for epoch in range(num_epochs): 

    running_loss = 0.0
    for i, samples in enumerate(train_loader, 0): 
        inputs, labels = samples

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = tdnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if epoch % 25 == 0:  
        print(f'{epoch} loss: {running_loss/(len(data) - test_length)}')
        losses.append(running_loss/(len(data) - test_length))
        running_loss = 0.0

print('Finished Training', num_epochs, 'epochs')

plt.plot(losses)
plt.title("Training Loss on " + str(num_epochs) + " epochs")
plt.show()

'''
Testing
'''
print("Testing on", len(data) - test_length, "inputs (training data)")

correct = 0
total = 0
with torch.no_grad():
    for samples in train_loader:
        inputs, labels = samples
        outputs = tdnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the train samples: {100 * correct // total} %')

print("Testing on", test_length, "inputs")

correct = 0
total = 0
with torch.no_grad():
    for samples in test_loader:
        inputs, labels = samples
        outputs = tdnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test samples: {100 * correct // total} %')


'''
Saving Parameters
'''
model_params_path = 'model_params/model_params_004'
# torch.save({
#     'epoch': epoch,
#     'model_state_dict': tdnn.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss,
#     }, model_params_path)
# torch.save({'model': tdnn,
#             'mean': mean,
#             'std': std}, model_params_path)
# print('model parameters loaded into', model_params_path)
