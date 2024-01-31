import librosa
import numpy as np

np.float = float 

# read file
file    = "../15322/project1/comp.wav"
sig, fs = librosa.core.load(file, sr=8000)

# process
abs_spectrogram = np.abs(librosa.core.spectrum.stft(sig))
audio_signal = librosa.core.spectrum.griffinlim(abs_spectrogram)

print(audio_signal, audio_signal.shape)

# write output
librosa.output.write_wav('test2.wav', audio_signal, fs)