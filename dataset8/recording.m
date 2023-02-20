fs = 22050;
nBits = 16;
recobj = audiorecorder(fs, nBits, 1);
for i = 1:20
	recordblocking(recobj, 1);
	disp(int2str(i));
	word = getaudiodata(recobj);
	S = melSpectrogram(word, fs, 'NumBands', 16);
	filename = "ball" + int2str(i);
	writematrix(S, "spectrograms/b/" + filename + ".txt");
	audiowrite("wavs/b/" + filename + ".wav", word, fs);
end