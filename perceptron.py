import torch.nn as nn
from mytools import *
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy
import argparse

data = []
data_labels = []
files = []

def one_hot(matrix):
    shape = (matrix.size, matrix.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(matrix.size)
    one_hot[rows, matrix.ravel()] = 1
    one_hot = one_hot.reshape((*matrix.shape, matrix.max()+1))
    return one_hot

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--shift', dest='shift', type=int,
                        default=0, help="Number of frames to shift")
    parser.add_argument('--verbose', dest="verbose", type=bool,
                        default=False, help="print useless stuffs")
    return parser.parse_args()

args = parse_arguments()
load_dataset("dataset9/", data, data_labels, files, 9, 0)
load_dataset("dataset9/other_data/", data, data_labels, files, 9, 0)


len_train_data = len(data)
if args.verbose:
    for i in range(len(data)):
        for j in range(len(data[0])):
            if len(data[i][j]) != 15:
                print(len(data[i][j]), i, j, files[i])
data_tensor = torch.Tensor(data) # turn into torch tensor
eps = 10**-25 # avoid -inf in log
# logged_data = 10 * torch.log10(data_tensor + eps) 
# transformation based on matlab data (melspectrogram transformation for plotting)

data_tensor = torch.flatten(data_tensor, start_dim=1, end_dim=2)
# normalize
std = torch.std(data_tensor, 0)
mean = torch.mean(data_tensor, 0)
normed_data = (data_tensor - mean)/std # normalize 

# process labels and data loader
labels_tensor = torch.tensor(data_labels, dtype=torch.long)

labels_tensor = one_hot(labels_tensor.numpy())
print('labels have shape: ', labels_tensor.shape)
print('inputs have shape: ', normed_data.shape)

weights = np.dot(np.linalg.pinv(normed_data.numpy()), labels_tensor)

print('weights have shape: ', weights.shape)


'''
TESTING
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

data_tensor_test = torch.Tensor(data_test) # turn into torch tensor

data_tensor_test = torch.flatten(data_tensor_test, start_dim=1, end_dim=2)
# normalize
std_test = torch.std(data_tensor_test, 0)
mean_test = torch.mean(data_tensor_test, 0)
normed_data_test = (data_tensor_test - mean_test)/std_test # normalize 

# process labels and data loader
labels_tensor_test = torch.tensor(data_labels_test, dtype=torch.long)

labels_tensor_test = one_hot(labels_tensor_test.numpy())
print('labels have shape: ', labels_tensor_test.shape)
print('inputs have shape: ', normed_data_test.shape)

y_pred = np.dot(weights.T, normed_data_test.numpy().T)
print('y_preds have shape: ', y_pred.shape)
display_matrix(y_pred)
display_matrix(labels_tensor_test)

# add filespaths to dataset
# a temporary list to store the string labels
# files_dict = {}
# files_int_labels = []
# for i in range(len(files)):
#     files_dict[i] = files[i]
#     files_int_labels.append(i)

# files_tensor = torch.tensor(files_int_labels)
# whole_dataset = TensorDataset(normed_data, labels_tensor, files_tensor)

# train_loader = DataLoader(whole_dataset, shuffle=True) # create dataloader object


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.10, momentum=0.2, weight_decay=0.01)

# # hyperparameters
# num_epochs = 500
# losses = []

# for epoch in range(num_epochs): 

#     running_loss = 0.0
#     for i, samples in enumerate(train_loader, 0): 
#         inputs, labels, _ = samples

#         # zero the parameter gradients
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#     if epoch % 25 == 0:  
#         print(f'{epoch} loss: {running_loss/len_train_data}')
#         losses.append(running_loss/len_train_data)
#         running_loss = 0.0
#     print('Finished Training', num_epochs, 'epochs')

#     plt.plot(losses)
#     plt.title("Training Loss on " + str(num_epochs) + " epochs")
#     plt.show()