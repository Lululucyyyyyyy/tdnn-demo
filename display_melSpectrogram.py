import matplotlib.pyplot as plt
import numpy as np
import math
import torch

spectrogram_path = 'dataset5/spectrograms/d/dance00001-430.txt'

def display(path):
	with open(path, 'r') as f:
		spectrogram = [[float(num) for num in line.split(',')] for line in f]
	f.close()

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