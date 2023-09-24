import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import ast
import argparse

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-f', type=str, help='spectrogram file to be parsed')
parser.add_argument('-a', type=bool, help='is it in list format')

args = parser.parse_args()
spectrogram_path = args.f
already_list = args.a


def display(path):
	spectrogram = []
	with open(path, 'r') as f:
		if (already_list):
			spectrogram = ast.literal_eval(f.read())
		else:
			spectrogram = [[float(num) for num in line.split(',')] for line in f]
	f.close()

	print(spectrogram)
	spectrogram_np = np.array(spectrogram)
	melSpectrogram_np = np.log(spectrogram_np)

	fig = plt.figure(figsize=(3, 3))
	ax = fig.add_subplot(111)
	ax.set_title('melSpectrogram')
	plt.imshow(melSpectrogram_np, origin = 'lower', aspect='auto')
	ax.set_aspect(aspect='auto')

	cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
	cax.get_xaxis().set_visible(False)
	cax.get_yaxis().set_visible(False)
	cax.patch.set_alpha(0)
	cax.set_frame_on(False)
	plt.colorbar(orientation='vertical')
	plt.show()

def display_matrix(melSpectrogram):
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

display(spectrogram_path)