import os, re

def rename_file(file, compare=True):
	# extract start time of initial word segment in msec
	with open(file, 'r') as f:
		melSpectrogram = [[float(num) for num in line.split(',')] for line in f]
	f.close()

	new_parts = file.split('/')
	part1 = '/'.join(new_parts[:-1]) + '/'
	nameanddash = new_parts[-1].split('.')[0]
	name = nameanddash.split('-')[0]
	parts = re.split('(\d+)', name)
	name = parts[0]
	snum = parts[1]
	for i in range(5 - len(str(snum))):
		snum = "0" + snum
	num = nameanddash.split('-')[1]
	txt = new_parts[-1].split('.')[1]
	new_name = part1 + name + snum + "-" + num+ "." + txt
	os.rename(file, new_name)
	print(new_name)

def rename_dataset(file_path):
	for f in sorted(os.listdir(file_path + "b")):
		rename_file(file_path+"b/" + f, compare=True)
    
	for f in sorted(os.listdir(file_path + "d")):
		rename_file(file_path+"d/" + f, compare=True)
       
	for f in sorted(os.listdir(file_path + "g")):
		rename_file(file_path+"g/" + f, compare=True)

# rename_dataset("dataset3/spectrograms/")
# rename_dataset("dataset4/spectrograms/")
# rename_dataset("dataset5/spectrograms/")