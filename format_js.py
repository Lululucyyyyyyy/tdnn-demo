param_path = 'model_params/013/'

with open('params.js', 'w+') as f:
    f.write("const DEBUG = true; \n")
    f.write("/** ===================================================== \n *            PARAMETERS  \n* ===================================================== \n*/")
    f.write("\n")
    f.write("\n")
    f.write("var tdnn1Weight = [")
    with open(param_path + 'tdnn1.temporal_conv.weight.txt', 'r') as g:
        lines = g.readlines()
        for i in range(len(lines) - 1):
            f.write("[" + lines[0].split('\n')[0] + "], \n")
        f.write("[" + lines[len(lines) - 1].split('\n')[0] + "] ];")
        g.close()
    f.write("\n")
    f.write("\n")
    f.write("var tdnn1Bias = [")
    with open(param_path + 'tdnn1.temporal_conv.bias.txt', 'r') as g:
        lines = g.readlines()
        for i in range(len(lines) - 1):
            f.write(lines[0].split('\n')[0] + ", \n")
        f.write(lines[len(lines) - 1].split('\n')[0] + "];")
        g.close()
    f.write("\n")
    f.write("\n")
    f.write("var tdnn2Weight = [")
    with open(param_path + 'tdnn2.temporal_conv.weight.txt', 'r') as g:
        lines = g.readlines()
        for i in range(len(lines) - 1):
            f.write("[" + lines[0].split('\n')[0] + "], \n")
        f.write("[" + lines[len(lines) - 1].split('\n')[0] + "] ];")
        g.close()
    f.write("\n")
    f.write("\n")
    f.write("var tdnn2Bias = [")
    with open(param_path + 'tdnn2.temporal_conv.bias.txt', 'r') as g:
        lines = g.readlines()
        for i in range(len(lines) - 1):
            f.write(lines[0].split('\n')[0] + ", \n")
        f.write(lines[len(lines) - 1].split('\n')[0] + "];")
        g.close()
    f.write("\n")
    f.write("\n")
    f.write("var linearWeight = [")
    with open(param_path + 'linear.weight.txt', 'r') as g:
        lines = g.readlines()
        for i in range(len(lines) - 1):
            f.write("[" + lines[0].split('\n')[0] + "], \n")
        f.write("[" + lines[len(lines) - 1].split('\n')[0] + "] ];")
        g.close()
    f.write("\n")
    f.write("\n")
    f.write("var linearBias = [")
    with open(param_path + 'linear.bias.txt', 'r') as g:
        lines = g.readlines()
        for i in range(len(lines) - 1):
            f.write(lines[0].split('\n')[0] + ", \n")
        f.write(lines[len(lines) - 1].split('\n')[0] + "];")
        g.close()
    f.write("\n")
    f.write("\n")
    f.write("\n")
    f.write("var mean = tf.tensor([")
    with open(param_path + 'mean.txt', 'r') as g:
        lines = g.readlines()
        for i in range(len(lines) - 1):
            f.write("[" + lines[0].split('\n')[0] + "], \n")
        f.write("[" + lines[len(lines) - 1].split('\n')[0] + "] ]);")
        g.close()
    f.write("\n")
    f.write("\n")
    f.write("var std = tf.tensor([")
    with open(param_path + 'std.txt', 'r') as g:
        lines = g.readlines()
        for i in range(len(lines) - 1):
            f.write("[" + lines[0].split('\n')[0] + "], \n")
        f.write("[" + lines[len(lines) - 1].split('\n')[0] + "] ]);")
        g.close()
    f.write("\n")
    f.write("\n")
