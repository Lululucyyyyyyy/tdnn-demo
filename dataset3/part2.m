filename = "bird1-220";
writematrix(S, "spectrograms/b/" + filename + ".txt");
audiowrite("wavs/b/" + filename + ".wav", word, fs);