'''
functions:

extract_start_time: compares/renames file based on start time, or onset of word
align_dataset: compares or renames an entire dataset
load_file: helper function that loads a file to data, and its label to data_labels
load_null_sample: helper function extracts null samples and loads to data and data_labels
display: given a path to file, displays the spectrogram
display_matrix: given a spectrogram, displays it
unname: unnames a file
unname_dataset: unnames files in dataset
name: names all files in file_path to word + i
'''
import numpy as np
import os
from matplotlib import pyplot as plt
import re
import random
import math
import torch
import ast
from os.path import isfile, join
import torch.nn as nn
from tdnn import TDNN as TDNNLayer

# file renaming threshold
thresh = 0.1

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

def extract_start_time(file, compare=True):
	'''
	if compare is True, only compares file's predicted starttime and starttime in filename
	if false, renames file to contain starttime in the format: wordnum-starttime.txt.
        Example: ba00001-300.txt
	'''
	# extract start time of initial word segment in msec
	if not file.endswith(".txt"):
		return

	with open(file, 'r') as f:
		melSpectrogram = [[float(num) for num in line.split(',')] for line in f]
	f.close()
	melSpectrogram = np.array(melSpectrogram)
	melSpectrogram = np.exp(melSpectrogram)

	# alignment
	col_sums = np.sum(melSpectrogram, axis=0)
	col_sums[0:10] = 0# mic onset skews data
	max_col_sum = max(col_sums)
	col_sums_normed = col_sums / max_col_sum
	col_sums_normed_np = np.array(col_sums_normed)
	thresh_indexes = np.argwhere(col_sums_normed_np > thresh)
	start_time_ms = thresh_indexes[0]*10 - 20 # to capture onset in msec

	if compare:
	# comparison
		curr_time_ms = int((file.split('-')[-1]).split('.')[0])
		if abs(start_time_ms - curr_time_ms) > 30:
			print(file, start_time_ms, curr_time_ms)
			x_coords = [i*10 for i in range(len(col_sums_normed))]
			plt.plot(x_coords, col_sums_normed)
			plt.plot([start_time_ms] * 2, [0, 1])
			plt.plot([curr_time_ms] * 2, [0, 1], 'g')
			plt.show()
	else:
		new_parts = file.split('/')
		part1 = '/'.join(new_parts[:-1]) + '/'
		name = new_parts[-1].split('.')[0]
		parts = re.split('(\d+)', name)
		name = parts[0]
		snum = int(parts[1])
		snum = "{:05d}".format(snum)
		txt = new_parts[-1].split('.')[1]
		new_name = part1 + name + snum + "-" + str(start_time_ms[0]) + "." + txt
		os.rename(file, new_name)
	return start_time_ms

def align_dataset(file_path, comp=True):
	'''
	aligns entire dataset
	'''
	for f in sorted(os.listdir(file_path + "b")):
		extract_start_time(file_path+"b/" + f, compare=comp)
    
	for f in sorted(os.listdir(file_path + "d")):
		extract_start_time(file_path+"d/" + f, compare=comp)
       
	for f in sorted(os.listdir(file_path + "g")):
		extract_start_time(file_path+"g/" + f, compare=comp)

def load_file(f, label, letter, path_to_training_data, data, data_labels, files):
	'''
	helper function that loads a file to data, and its label to data_labels
	inputs:
	- f: the filename
	- label: the class, can be 0(b), 1(d), or 2(g)
	- path_to_training_data: the path where the file is
	- data: the list where the data is stored
	- data_labels: the list wehre the labels are stored
	- files: the list where the filenames are stored
	'''
	if not f.endswith(".txt"):
		return
	fs = f.split("-") # split file name to get starting frame
	start = int(fs[1].split('.')[0])//10
	with open(path_to_training_data+letter+"/"+f, 'r') as g:
		mel_spectrogram = [[float(num) for num in line.split(',')][start:start + 15] for line in g]
		data.append(mel_spectrogram)
		files.append(f)
		if (np.array(mel_spectrogram).shape != (16, 15)):
			print(f, np.array(mel_spectrogram).shape)
	g.close()
	data_labels.append(label)

def load_file_shifted(f, label, letter, path_to_training_data, data, data_labels, files, num_frames_shifted):
	'''
	helper function that loads a file to data, but slightly shifted, and its label to data_labels
	inputs:
	- f: the filename
	- label: the class, can be 0(b), 1(d), or 2(g)
	- path_to_training_data: the path where the file is
	- data: the list where the data is stored
	- data_labels: the list wehre the labels are stored
	- files: the list where the filenames are stored
	'''
	if not f.endswith(".txt"):
		return
	fs = f.split("-") # split file name to get starting frame
	start = int(fs[1].split('.')[0])//10 + num_frames_shifted
	broken = False
	with open(path_to_training_data+letter+"/"+f, 'r') as g:
		mel_spectrogram = []
		for line in g:
			if start + 15 >= len(line.split(',')):
				broken = True
				break
			else:
				mel_spectrogram.append([float(num) for num in line.split(',')][start:start + 15])
		if not broken:
			data.append(mel_spectrogram)
			data_labels.append(label)
			files.append(f)
	g.close()

def load_null_sample(f, letter, path_to_training_data, data, data_labels, files):
	'''
	helper function extracts null samples and loads to data and data_labels
	the datalabel is 4
	inputs:
	- f: the filename
	- letter: b, d, or g=
	- path_to_training_data: the path where the file is
	- data: the list where the data is stored
	- data_labels: the list wehre the labels are stored
	- files: the list where the filenames are stored
	'''
	
	if not f.endswith(".txt"):
		return
	fs = f.split("-") # split file name to get starting frame
	start = int(fs[1].split('.')[0])//10

    # start time extractions for null files
    # if start > 200, then 0 - 150, 50 - 200
    # start + 500 onwards, 150 ms each, 50 ms apart
    # start + 100 ms
	with open(path_to_training_data+letter+"/"+f, 'r') as g:
		mel_spectrogram = [[float(num) for num in line.split(',')] for line in g]
	if start > 20:
		data.append([line[0:15] for line in mel_spectrogram])
		data.append([line[5:20] for line in mel_spectrogram])
		data_labels.append(3)
		data_labels.append(3)
		files.append(f + 'null: ' + str(0))
		files.append(f + 'null: ' + str(5))
	s = start + 50
	while s + 15 < 97:
		data.append([line[s:s+15] for line in mel_spectrogram])
		data_labels.append(3)
		files.append(f + 'null: ' + str(s))
		s += 50
	if start + 25 < 97:
		data.append([line[start+10:start+25] for line in mel_spectrogram])
		data_labels.append(3)
		files.append(f + 'null: ' + str(start + 10))
	g.close()

def load_dataset(path_to_training_data, data, data_labels, files, dataset_num, prune_null):
	'''
	Uses load_file and load_null_sample to load an entire dataset

	inputs:
	- path_to_training_data: the path where the file is
	- data: the list where the data is stored
	- data_labels: the list wehre the labels are stored
	- files: the list where the filenames are stored
	'''
	l_before = len(data)
	print("loading", path_to_training_data)
	for f in sorted(os.listdir(path_to_training_data + "b")):
		load_file(f, 0, "b", path_to_training_data, data, data_labels, files)
    
	for f in sorted(os.listdir(path_to_training_data + "d")):
		load_file(f, 1, "d", path_to_training_data, data, data_labels, files)
       
	for f in sorted(os.listdir(path_to_training_data + "g")):
		load_file(f, 2, "g", path_to_training_data, data, data_labels, files)

	length = len(data) - l_before

	if prune_null != 0:
		null_data = []
		null_data_labels = []
		null_files = []

		for f in sorted(os.listdir(path_to_training_data + "b")):
			load_null_sample(f, "b", path_to_training_data, null_data, null_data_labels, null_files)

		for f in sorted(os.listdir(path_to_training_data + "d")):
			load_null_sample(f, "d", path_to_training_data, null_data, null_data_labels, null_files)

		for f in sorted(os.listdir(path_to_training_data + "g")):
			load_null_sample(f, "g", path_to_training_data, null_data, null_data_labels, null_files)
		
		num_null = int(len(null_data)*prune_null)
		nd_sub, ndl_sub, nf_sub = zip(*random.sample(list(zip(null_data, null_data_labels, null_files)), num_null))

		data += nd_sub
		data_labels += ndl_sub
		files += nf_sub

    # for f in sorted(os.listdir(path_to_training_data + "b")):
    #     load_null_sample(f, "b", path_to_training_data, data, data_labels, files, dataset_num)

    # for f in sorted(os.listdir(path_to_training_data + "d")):
    #     load_null_sample(f, "d", path_to_training_data, data, data_labels, files, dataset_num)

    # for f in sorted(os.listdir(path_to_training_data + "g")):
    #     load_null_sample(f, "g", path_to_training_data, data, data_labels, files, dataset_num)

	print("loaded", length, "samples (not including null files)")
	length = len(data) - l_before
	print("loaded", length, "samples (including null files)")

def load_dataset_only_b(path_to_training_data, data, data_labels, files, dataset_num, prune_null):
	'''
	Uses load_file to load an entire dataset only for b

	inputs:
	- path_to_training_data: the path where the file is
	- data: the list where the data is stored
	- data_labels: the list wehre the labels are stored
	- files: the list where the filenames are stored
	'''
	l_before = len(data)
	print("loading", path_to_training_data)
	for f in sorted(os.listdir(path_to_training_data + "b")):
		load_file(f, 0, "b", path_to_training_data, data, data_labels, files)

	length = len(data) - l_before
	print("loaded", length, "samples (not including null files)")

def load_dataset_shifted(path_to_training_data, data, data_labels, files, dataset_num, prune_null, shift_frames):
	'''
	Uses load_file and load_null_sample to load an entire dataset, with shifted frames

	inputs:
	- path_to_training_data: the path where the file is
	- data: the list where the data is stored
	- data_labels: the list wehre the labels are stored
	- files: the list where the filenames are stored
	'''
	l_before = len(data)
	print("loading", path_to_training_data)
	for f in sorted(os.listdir(path_to_training_data + "b")):
		load_file_shifted(f, 0, "b", path_to_training_data, data, data_labels, files, shift_frames)
    
	for f in sorted(os.listdir(path_to_training_data + "d")):
		load_file_shifted(f, 1, "d", path_to_training_data, data, data_labels, files, shift_frames)
       
	for f in sorted(os.listdir(path_to_training_data + "g")):
		load_file_shifted(f, 2, "g", path_to_training_data, data, data_labels, files, shift_frames)

	length = len(data) - l_before

	print("loaded", length, "samples (not including null files)")
	length = len(data) - l_before
	print("loaded", length, "samples (including null files)")

def display(path):
	'''
	given a path to file, displays the spectrogram
	'''
	transpose = False
	already_list = False
	spectrogram = []
	with open(path, 'r') as f:
		if (already_list):
			spectrogram = ast.literal_eval(f.read())
		else:
			spectrogram = [[float(num) for num in line.split(',')] for i, line in enumerate(f) if i != 0]
	f.close()

	# print(spectrogram)
	spectrogram_np = np.array(spectrogram)
	if transpose:
		spectrogram_np = spectrogram_np.T
	eps = 10**-25 # avoid log error
	#spectrogram_np = np.log(spectrogram_np + eps)
	# print(melSpectrogram_np)

	fig = plt.figure(figsize=(3, 3))
	ax = fig.add_subplot(111)
	ax.set_title('melSpectrogram')
	plt.imshow(spectrogram_np, origin = 'lower', aspect='auto')
	ax.set_aspect(aspect='auto')

	cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
	cax.get_xaxis().set_visible(False)
	cax.get_yaxis().set_visible(False)
	cax.patch.set_alpha(0)
	cax.set_frame_on(False)
	plt.colorbar(orientation='vertical')
	plt.show()

def display_matrix(melSpectrogram):
	'''
	given a spectrogram, displays it
	'''
	melSpectrogram_np = np.array(melSpectrogram)
	fig = plt.figure(figsize=(3, 3))
	ax = fig.add_subplot(111)
	ax.set_title('melSpectrogram')
	plt.imshow(melSpectrogram_np, aspect='auto')
	ax.set_aspect('auto')

	cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
	cax.get_xaxis().set_visible(False)
	cax.get_yaxis().set_visible(False)
	cax.patch.set_alpha(0)
	cax.set_frame_on(False)
	plt.colorbar(orientation='vertical')
	plt.show()

def unname(file):
	'''
	unnames a file
	'''
	new_parts = file.split('/')
	part1 = '/'.join(new_parts[:-1]) + '/'
	name = new_parts[-1].split('-')[0][:-4]
	new_name = part1 + name +  "." + "txt"
	os.rename(file, new_name)

def unname_dataset(file_path, comp=True):
	'''
	unnames files in dataset
	'''
	for f in sorted(os.listdir(file_path + "b")):
		unname(file_path+"b/" + f, compare=comp)
    
	for f in sorted(os.listdir(file_path + "d")):
		unname(file_path+"d/" + f, compare=comp)
       
	for f in sorted(os.listdir(file_path + "g")):
		unname(file_path+"g/" + f, compare=comp)
		
def name(file_path, word, start):
	'''
	names all files in file_path to word + i
	'''
	for i, f in enumerate(sorted(os.listdir(file_path))):
		file_name = word + str(start + i) + ".txt"
		os.rename(file_path + f, file_path + file_name)

def display_both(path):
	'''
	given a path to file, displays the spectrogram and predictions
	'''
	transpose = False
	already_list = False
	spectrogram = []
	with open(path, 'r') as f:
		if (already_list):
			spectrogram = ast.literal_eval(f.read())
		else:
			spectrogram = [[float(num) for num in line.split(',')] for i, line in enumerate(f) if i != 0]
	f.close()

	# print(spectrogram)
	spectrogram_np = np.array(spectrogram)
	if transpose:
		spectrogram_np = spectrogram_np.T
	eps = 10**-25 # avoid log error
	#spectrogram_np = np.log(spectrogram_np + eps)
	# print(melSpectrogram_np)

	fig = plt.figure(figsize=(9, 7))
	ax = fig.add_subplot(2, 1, 1)
	ax.set_title('melSpectrogram')
	plt.imshow(spectrogram_np, origin = 'lower', aspect='auto')
	ax.set_aspect(aspect='auto')

	cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
	cax.get_xaxis().set_visible(False)
	cax.get_yaxis().set_visible(False)
	cax.patch.set_alpha(0)
	cax.set_frame_on(False)
	plt.colorbar(orientation='vertical')

	model_params_path = 'model_params/model_params_023'
	model_params = torch.load(model_params_path)
	model = model_params['model']
	mean = model_params['mean']
	std = model_params['std']

	spectrogram_file_path = path
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
	output_b += [0] * 15
	output_d += [0] * 15
	output_g += [0] * 15
	ax = fig.add_subplot(2, 1, 2)
	
	ax.set_title('melSpectrogram of ' + file_name)

	line1, = ax.plot(output_b, color="skyblue", label="b")	
	line2, = ax.plot(output_d, color="mediumslateblue", label="d")	
	line3, = ax.plot(output_g, color="plum", label="g")	
	# line4, = ax.plot(output_null, color="red", label="null")	
	plt.plot([time] * 2, [0, 1], color="khaki")

	ax.legend([line1, line2, line3], ["b", "d", "g"])
	plt.show()


# name("dataset9/naming/", "bait", 0)
# for f in sorted(os.listdir("dataset9/naming/")):
# 	extract_start_time("dataset9/naming/" + f, compare=False)

# align_dataset("dataset9/")

