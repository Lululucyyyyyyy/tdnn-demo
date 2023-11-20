from tdnn import TDNN as TDNNLayer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import math, random
import numpy as np
import os, re
import torch.optim as optim
import matplotlib.pyplot as plt
from mytools import load_dataset
from sklearn.metrics import classification_report
from mytools import *
import argparse

torch.manual_seed(0)

class TDNNv2(nn.Module):
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

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--shift', dest='shift', type=int,
                        default=0, help="Number of frames to shift")
    parser.add_argument('--verbose', dest="verbose", type=bool,
                        default=False, help="print useless stuffs")
    return parser.parse_args()

args = parse_arguments()

def load_and_process_data():
    '''
    Training Data set: 
    - labels 0 = b, 1 = d, 2 = g
    '''
    data = []
    data_labels = [] 
    files = []

    load_dataset("dataset9/", data, data_labels, files, 9, 0)
    load_dataset("dataset9/other_data/", data, data_labels, files, 9, 0)
    load_dataset_shifted("dataset9/", data, data_labels, files, 9, 0, args.shift)
    load_dataset_shifted("dataset9/other_data/", data, data_labels, files, 9, 0, args.shift)

    assert(len(data) == len(data_labels) and len(data) == len(files))

    '''
    Data Processing
    '''
    
    # test_length = int(len(data) * 0.1)
    # train_length = len(data) - test_length
    train_length = len(data)
    data_tensor = torch.Tensor(data) # turn into torch tensor
    eps = 10**-25 # avoid -inf in log
    # logged_data = 10 * torch.log10(data_tensor + eps) 
    # transformation based on matlab data (melspectrogram transformation for plotting)

    # normalize
    std = torch.std(data_tensor, 0)
    mean = torch.mean(data_tensor, 0)
    normed_data = (data_tensor - mean)/std # normalize 

    # process labels and data loader
    labels_tensor = torch.tensor(data_labels, dtype=torch.long)

    # add filespaths to dataset
    # a temporary list to store the string labels
    files_dict = {}
    files_int_labels = []
    for i in range(len(files)):
        files_dict[i] = files[i]
        files_int_labels.append(i)

    files_tensor = torch.tensor(files_int_labels)
    whole_dataset = TensorDataset(normed_data, labels_tensor, files_tensor)

    # train_dataset, test_dataset = torch.utils.data.random_split(whole_dataset, [train_length, test_length])
    batch_size = len(data) // 10
    num_batches = train_length // batch_size
    train_loader = DataLoader(whole_dataset, batch_size=batch_size, shuffle=True) # create dataloader object

    print("dataset done loading; loaded total of ", len(data), "samples")
    print(sum([0 if x != 0 else 1 for x in data_labels]), "b's, label:0")
    print(sum([0 if x != 1 else 1 for x in data_labels]), "d's, label:1")
    print(sum([0 if x != 2 else 1 for x in data_labels]), "g's, label:2")
    print(sum([0 if x != 3 else 1 for x in data_labels]), "null vals, label:3")
    print("number of training examples:", train_length)
    print("batch size:", batch_size, ", number of batches:", num_batches)

    '''
    testset is loaded separately
    '''
    data_test = []
    data_labels_test = []
    files_test = []
    if args.shift != 0:
        load_dataset_shifted("dataset9/", data_test, data_labels_test, files_test, 9, 0, args.shift)
        load_dataset_shifted("dataset9/other_data/", data_test, data_labels_test, files_test, 9, 0, args.shift)
    else:
        load_dataset("dataset9/", data_test, data_labels_test, files_test, 9, 0)
        load_dataset("dataset9/other_data/", data_test, data_labels_test, files_test, 9, 0)

    test_length = len(data_test)
    data_tensor_test = torch.Tensor(data_test) # turn into torch tensor
    #data_tensor_test = torch.flatten(data_tensor_test, start_dim=1, end_dim=2)
    # normalize
    std_test = torch.std(data_tensor_test, 0)
    mean_test = torch.mean(data_tensor_test, 0)
    normed_data_test = (data_tensor_test - mean_test)/std_test # normalize 

    files_dict = {}
    files_int_labels = []
    for i in range(len(files)):
        files_dict[i] = files[i]
        files_int_labels.append(i)

    files_tensor_test = torch.tensor(files_int_labels)

    # process labels and data loader
    labels_tensor_test = torch.tensor(data_labels_test, dtype=torch.long)
    test_dataset = TensorDataset(normed_data_test, labels_tensor_test, files_tensor_test)
    test_loader = DataLoader(test_dataset, batch_size=test_length, shuffle=True)
    return train_loader, test_loader, train_length, test_length, mean, std, files_dict

def train(train_loader, len_train_data):
    '''
    Training
    '''
    global tdnn
    tdnn = TDNNv2()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(tdnn.parameters(), lr=10e-2, momentum=0.3) #weight_decay=0.01

    # hyperparameters
    num_epochs = 500
    losses = []

    for epoch in range(num_epochs): 

        running_loss = 0.0
        for i, samples in enumerate(train_loader, 0): 
            inputs, labels, _ = samples

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = tdnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % 25 == 0:  
            print(f'{epoch} loss: {running_loss/len_train_data}')
            losses.append(running_loss/len_train_data)
            running_loss = 0.0

    print('Finished Training', num_epochs, 'epochs')

    plt.plot(losses)
    plt.title("Training Loss on " + str(num_epochs) + " epochs")
    plt.show()
    return tdnn, optimizer, loss, num_epochs

def test(tdnn_test, test_loader, train_length, test_length, save_incorrect, incorect_examples_path, files_dict = {}):
    '''
    Testing
    '''
    print("Testing on", train_length, "inputs (training data)")

    tdnn_test.eval()

    correct = 0
    total = 0
    f = open(incorect_examples_path, "a+")
    with torch.no_grad():
        for samples in train_loader:
            inputs, labels, files = samples
            outputs = tdnn_test(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if save_incorrect == True:
                for i, x in enumerate(predicted):
                    if x != labels[i]:
                        file_int = files[i].item()
                        file = files_dict[file_int]
                        f.write(file + ' ' + 'predicted: ' + str(x.item()) + '\n')

    print(f'Accuracy of the network on the train samples: {100 * correct // total} %')

    print("Testing on", test_length, "inputs")

    f.write('================ \n')

    correct = 0
    total = 0
    with torch.no_grad():
        for samples in test_loader:
            inputs, labels, files = samples
            outputs = tdnn_test(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #print(classification_report(labels, predicted))
            if save_incorrect == True:
                for i, x in enumerate(predicted):
                    if x != labels[i]:
                        file_int = files[i].item()
                        file = files_dict[file_int]
                        f.write(file + ' ' + 'predicted: ' + str(x.item()) + '\n')
    f.close()

    print(f'Accuracy of the network on the test samples: {100 * correct // total} %')
    if save_incorrect:
        print('incorrect examples loaded into', incorect_examples_path)

def save_params(tdnn, state_dict, optimizer, num_epochs, loss, mean, std, model_params_path):
    '''
    Saving Parameters
    '''
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': tdnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'state_dict': state_dict,
        }, model_params_path)
    torch.save({'model': tdnn,
                'mean': mean,
                'std': std}, model_params_path)
    print('model parameters loaded into', model_params_path)


train_loader, test_loader, len_train_data, len_test_data, mean, std, files_dict = load_and_process_data()

trained_tdnn, optimizer, num_epochs, loss = train(train_loader, 
                                                len_train_data)

test(trained_tdnn, 
    test_loader, 
    len_train_data, 
    len_test_data, 
    save_incorrect = False, 
    incorect_examples_path = 'incorrect_examples/samples_015', 
    files_dict = files_dict)

save_params(trained_tdnn, 
            trained_tdnn.state_dict(),
            optimizer, 
            num_epochs, 
            loss, 
            mean, 
            std, 
            model_params_path = 'model_params/model_params_015')


