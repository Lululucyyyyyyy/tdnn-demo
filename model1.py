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
from dataset_loader import load_dataset
from sklearn.metrics import classification_report

class TDNNv1(nn.Module):
    '''
    TDNN Model from Paper, consisting of the following layers:
    - tdnn 1: 16 in channels, 8 out channels, 15 samples, window of 3
    - sigmoid after tdnn
    - tdnn 2: 8 in channels, 3 out channels, 13 samples, window of 5
    - sigmoid after tdnn
    - flatten: 9 frequencies, 4 out channels, flattens to (27, ) array
    '''

    def __init__(self):
        super(TDNNv1, self).__init__()

        self.tdnn1 = TDNNLayer(16, 8, [-1,0,1])
        self.sigmoid1 = nn.Sigmoid()
        self.tdnn2 = TDNNLayer(8, 3, [-2,0,2])
        self.sigmoid2 = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(27, 4)
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

def load_and_process_data():
    '''
    Training Data set: 
    - labels 0 = b, 1 = d, 2 = g
    '''
    data = []
    data_labels = [] 
    files = []

    load_dataset("dataset3/spectrograms/", data, data_labels, files, '3')
    load_dataset("dataset4/spectrograms/", data, data_labels, files, '4')
    load_dataset("dataset5/spectrograms/", data, data_labels, files, '5')
    load_dataset("dataset6/spectrograms/", data, data_labels, files, '6')
    load_dataset("dataset7/spectrograms/", data, data_labels, files, '7')
    # load_dataset("dataset8/spectrograms/", data, data_labels, files, '8')

    '''
    Data Processing
    '''
    test_length = int(len(data) * 0.1)
    train_length = len(data) - test_length
    data_tensor = torch.Tensor(data) # turn into torch tensor
    logged_data = torch.log(data_tensor).max(torch.tensor(-25)) 
    # # take the log, element-wise, cap at -25 to avoid -inf

    # normalize
    std = torch.std(logged_data, 0)
    mean = torch.mean(logged_data, 0)
    normed_data = (logged_data - mean)/std # normalize 

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

    train_dataset, test_dataset = torch.utils.data.random_split(whole_dataset, [train_length, test_length])
    batch_size = 50
    num_batches = train_length // batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # create dataloader object

    print("dataset done loading; loaded total of ", len(data), "samples")
    print(sum([0 if x != 0 else 1 for x in data_labels]), "b's, label:0")
    print(sum([0 if x != 1 else 1 for x in data_labels]), "d's, label:1")
    print(sum([0 if x != 2 else 1 for x in data_labels]), "g's, label:2")
    print(sum([0 if x != 3 else 1 for x in data_labels]), "null vals, label:3")
    print("number of training examples:", len(data) - test_length)
    print("batch size:", batch_size, ", number of batches:", num_batches)

    '''
    testset is 10%
    '''
    test_loader = DataLoader(test_dataset, batch_size=test_length, shuffle=True)
    return train_loader, test_loader, train_length, test_length, mean, std, files_dict

def train(train_loader, len_train_data):
    '''
    Training
    '''

    tdnn = TDNNv1()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(tdnn.parameters(), lr=0.15, momentum=0.2)

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

def test(tdnn, test_loader, train_length, test_length, save_incorrect, incorect_examples_path, files_dict = {}):
    '''
    Testing
    '''
    print("Testing on", train_length, "inputs (training data)")

    tdnn.eval()

    correct = 0
    total = 0
    f = open(incorect_examples_path, "a+")
    with torch.no_grad():
        for samples in train_loader:
            inputs, labels, files = samples
            outputs = tdnn(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if save_incorrect == True:
                difference = np.subtract(predicted, labels)
                not_equal = np.ma.nonzero(difference)[0]
                incorrect = not_equal
                if(len(not_equal) > 0):
                    print(incorrect)
                    for x in incorrect:
                        print(files_dict[x])
                        f.write(files_dict[x] + ' ' + '\n')

    print(f'Accuracy of the network on the train samples: {100 * correct // total} %')

    print("Testing on", test_length, "inputs")

    f.write('================ \n')

    correct = 0
    total = 0
    with torch.no_grad():
        for samples in test_loader:
            inputs, labels, files = samples
            outputs = tdnn(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(classification_report(labels, predicted))
            if save_incorrect == True:
                difference = np.subtract(predicted, labels)
                not_equal = np.ma.nonzero(difference)[0]
                incorrect = not_equal
                if(len(not_equal) > 0):
                    print(incorrect)
                    for x in incorrect:
                        print(files_dict[x])
                        f.write(files_dict[x] + ' ' + '\n')
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
    save_incorrect = True, 
    incorect_examples_path = 'incorrect_examples/samples_009', 
    files_dict = files_dict)

# save_params(trained_tdnn, 
#             optimizer, 
#             num_epochs, 
#             loss, 
#             mean, 
#             std, 
#             model_params_path = 'model_params/model_params_008')

save_params(trained_tdnn, 
            trained_tdnn.state_dict(),
            optimizer, 
            num_epochs, 
            loss, 
            mean, 
            std, 
            model_params_path = 'model_params/model_params_009')


