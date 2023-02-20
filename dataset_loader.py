import os
from display_melSpectrogram import display, display_matrix

def load_file(f, label, letter, path_to_training_data, data, data_labels, files, dataset_num):
    if not f.endswith(".txt"):
        return
    fs = f.split("-") # split file name to get starting frame
    start = int(fs[1].split('.')[0])//10
    with open(path_to_training_data+letter+"/"+f, 'r') as g:
        mel_spectrogram = [[float(num) for num in line.split(',')][start:start + 15] for line in g]
        data.append(mel_spectrogram)
        files.append(dataset_num + fs[0])
    g.close()
    data_labels.append(label)

def load_null_sample(f, letter, path_to_training_data, data, data_labels, files, dataset_num):
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
		files.append(dataset_num + fs[0])
		files.append(dataset_num + fs[0])
	s = start + 50
	while s + 15 < 97:
		data.append([line[s:s+15] for line in mel_spectrogram])
		data_labels.append(3)
		files.append(dataset_num + fs[0])
		s += 50
	if start + 25 < 97:
		data.append([line[start+10:start+25] for line in mel_spectrogram])
		data_labels.append(3)
		files.append(dataset_num + fs[0])
	g.close()


def load_dataset(path_to_training_data, data, data_labels, files, dataset_num):
    l_before = len(data)
    print("loading", path_to_training_data)
    for f in sorted(os.listdir(path_to_training_data + "b")):
        load_file(f, 0, "b", path_to_training_data, data, data_labels, files, dataset_num)
    
    for f in sorted(os.listdir(path_to_training_data + "d")):
        load_file(f, 1, "d", path_to_training_data, data, data_labels, files, dataset_num)
       
    for f in sorted(os.listdir(path_to_training_data + "g")):
        load_file(f, 2, "g", path_to_training_data, data, data_labels, files, dataset_num)

    length = len(data) - l_before

    for f in sorted(os.listdir(path_to_training_data + "b")):
        load_null_sample(f, "b", path_to_training_data, data, data_labels, files, dataset_num)

    for f in sorted(os.listdir(path_to_training_data + "d")):
        load_null_sample(f, "d", path_to_training_data, data, data_labels, files, dataset_num)

    for f in sorted(os.listdir(path_to_training_data + "g")):
        load_null_sample(f, "g", path_to_training_data, data, data_labels, files, dataset_num)

    print("loaded", length, "samples (not including null files)")
    length = len(data) - l_before
    print("loaded", length, "samples (including null files)")



