import numpy as np
import os
from matplotlib import pyplot as plt
import re

# file
thresh = 0.1

def extract_start_time(file, compare=True):
	# extract start time of initial word segment in msec
	if not file.endswith(".txt"):
		return

	with open(file, 'r') as f:
		melSpectrogram = [[float(num) for num in line.split(',')] for line in f]
	f.close()

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

def align_dataset(file_path, comp=False):
	for f in sorted(os.listdir(file_path + "b")):
		extract_start_time(file_path+"b/" + f, compare=comp)
    
	for f in sorted(os.listdir(file_path + "d")):
		extract_start_time(file_path+"d/" + f, compare=comp)
       
	for f in sorted(os.listdir(file_path + "g")):
		extract_start_time(file_path+"g/" + f, compare=comp)

# align_dataset("dataset3/spectrograms/")
# align_dataset("dataset4/spectrograms/")
# align_dataset("dataset5/spectrograms/")
