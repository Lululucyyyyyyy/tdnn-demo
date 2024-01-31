from tdnn import TDNN as TDNNLayer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from mytools import load_dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from mytools import *
import argparse

# torch.manual_seed(0)

class TDNNv2(nn.Module):
    '''
    TDNN Model from Paper, consisting of the following layers:
    - tdnn 1: 16 in channels, 8 out channels, 15 samples, window of 3
    - sigmoid after tdnn
    - tdnn 2: 8 in channels, 3 out channels, 13 samples, window of 5
    - sigmoid after tdnn
    - flatten: 9 frequencies, 3 out channels, flattens to (27, ) array
    - linear: 27 inputs, 4 outputs
    - sigmoid after final layer
    '''

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
            self.sigmoid3,
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
    parser.add_argument('--save', dest='save', type=bool,
                        default=False, help="save parameters>")
    parser.add_argument('--path_num', dest='path_num', type=int,
                        default=0, help="which model param path to save to")
    parser.add_argument('--show_plt', dest='show_plot', type=bool,
                        default=False, help="show training plots")
    return parser.parse_args()

args = parse_arguments()
num_epochs = 600
learning_rate = 0.3
momentum = 0.3
k_folds = 3
batch_size = 20

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
    # load_dataset("dataset10/allen/", data, data_labels, files, 10, 0)
    while args.shift != 0:
        # load_dataset("dataset10/allen/", data, data_labels, files, 10, args.shift)
        # load_dataset("dataset10/allen/", data, data_labels, files, 10, -args.shift)
        load_dataset_shifted("dataset9/", data, data_labels, files, 9, 0, args.shift)
        load_dataset_shifted("dataset9/other_data/", data, data_labels, files, 9, 0, args.shift)
        load_dataset_shifted("dataset9/", data, data_labels, files, 9, 0, -args.shift)
        load_dataset_shifted("dataset9/other_data/", data, data_labels, files, 9, 0, -args.shift)
        args.shift -= 1

    assert(len(data) == len(data_labels) and len(data) == len(files))

    '''
    Data Processing
    '''
    
    test_length = int(len(data) * 0.1)
    train_length = len(data) - test_length
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

    # split into K folds
    kfold = KFold(n_splits=k_folds, shuffle=True)

    train_dataset, test_dataset = torch.utils.data.random_split(whole_dataset, [train_length, test_length])

    print("dataset done loading; loaded total of ", len(data), "samples")
    print(sum([0 if x != 0 else 1 for x in data_labels]), "b's, label:0")
    print(sum([0 if x != 1 else 1 for x in data_labels]), "d's, label:1")
    print(sum([0 if x != 2 else 1 for x in data_labels]), "g's, label:2")
    print(sum([0 if x != 3 else 1 for x in data_labels]), "null vals, label:3")
    print('========== hyperparameters ============')
    print("number of training examples:", train_length)
    print("batch size:", batch_size)
    print("learning rate:", learning_rate, ", momentum:", momentum)
    print('epochs:', num_epochs)

    return train_length, test_length, mean, std, files_dict, kfold, train_dataset, test_dataset

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def train(train_dataset, kfold):
    '''
    Training
    '''
    results = {}
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold}')
        print('------------------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, sampler=train_subsampler)
        validloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, sampler=valid_subsampler
        )

        # global tdnn
        tdnn = TDNNv2()
        tdnn.apply(reset_weights)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(tdnn.parameters(), lr=learning_rate, momentum=momentum) #weight_decay=0.01
        losses = []

        # train current batch
        for epoch in range(num_epochs): 

            running_loss = 0.0
            for i, samples in enumerate(trainloader, 0): 
                inputs, labels, _ = samples
                labels = torch.nn.functional.one_hot(labels, num_classes=3).float()

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = tdnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            if epoch % 25 == 0:  
                print(f'{epoch} loss: {running_loss/25}')
                losses.append(running_loss/25)
                running_loss = 0.0

        print('Finished Training Fold', fold, 'with', num_epochs, 'epochs')
        if args.show_plot:
            plt.plot(losses)
            plt.title("Training Loss on " + str(num_epochs) + " epochs")
            plt.show()

        # evaluate current fold
        path_num = "{:03d}".format(args.path_num)
        save_path = f'model_params/folds/{path_num}_{fold}'
        torch.save(tdnn.state_dict(), save_path)

        correct, total = 0, 0
        with torch.no_grad():
            for i, data in enumerate(validloader, 0):
                inputs, targets, _ = data
                outputs = tdnn(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('----------------------------')
            results[fold] = 100.0 * (correct / total)

    # K fold results
    print(f'K-fold Cross Validation Results for {k_folds} Folds')
    print('--------------------------------')
    sum = 0.0
    best, best_val = 0, 0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value 
        if value > best_val:
            best = key
            best_val = value
    print(f'Average: {sum/len(results.items())} %')

    path_num = "{:03d}".format(args.path_num)
    best_path = f'model_params/folds/{path_num}_{best}'
    model = TDNNv2()
    model.load_state_dict(torch.load(best_path))

    return model, optimizer, loss, num_epochs

def test(tdnn_test, test_dataset, train_length, test_length, save_incorrect, incorect_examples_path, files_dict = {}):
    '''
    Testing
    '''
    print("Testing on", train_length, "inputs (training data)")

    tdnn_test.eval()

    correct = 0
    total = 0
    f = open(incorect_examples_path, "a+")
    # with torch.no_grad():
    #     for samples in train_loader:
    #         inputs, labels, files = samples
    #         outputs = tdnn_test(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         if save_incorrect == True:
    #             for i, x in enumerate(predicted):
    #                 if x != labels[i]:
    #                     file_int = files[i].item()
    #                     file = files_dict[file_int]
    #                     f.write(file + ' ' + 'predicted: ' + str(x.item()) + '\n')

    # print(f'Accuracy of the network on the train samples: {100 * correct // total} %')

    print("Testing on", test_length, "inputs")

    f.write('================ \n')

    testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size = len(test_dataset))
    correct = 0
    total = 0
    with torch.no_grad():
        for samples in testloader:
            inputs, labels, files = samples
            outputs = tdnn_test(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(classification_report(labels, predicted))
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


len_train_data, len_test_data, mean, std, files_dict, k_folded_data, train_dataset, test_dataset = load_and_process_data()

trained_tdnn, optimizer, num_epochs, loss = train(train_dataset,
                                                k_folded_data)

test(trained_tdnn, 
    test_dataset, 
    len_train_data, 
    len_test_data, 
    save_incorrect = args.save, 
    incorect_examples_path = 'incorrect_examples/samples_'+"{:03d}".format(args.path_num), 
    files_dict = files_dict)

save_params(trained_tdnn, 
            trained_tdnn.state_dict(),
            optimizer, 
            num_epochs, 
            loss, 
            mean, 
            std, 
            model_params_path = 'model_params/model_params_'+"{:03d}".format(args.path_num))


