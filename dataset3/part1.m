fs = 11025;
nBits = 16;
recobj = audiorecorder(fs, nBits, 1);
recordblocking(recobj, 1);
word = getaudiodata(recobj);
melSpectrogram(word, fs, 'NumBands', 16);
S = melSpectrogram(word, fs, 'NumBands', 16);
