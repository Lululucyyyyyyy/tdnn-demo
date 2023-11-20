from argparse import ArgumentParser

def parse_arguments():
    # Command-line flags are defined here.
    parser = ArgumentParser()
    parser.add_argument('--n', dest='n', type=int,
                        default=13, help="Model Number")
    parser.add_argument('--f', dest='file', type=str,
                        default="lolyouforgotthefile.js", help="Javascript File")
    parser.add_argument('--verbose', dest="verbose", type=bool,
                        default=False, help="print useless stuffs")
    return parser.parse_args()

args = parse_arguments()

# param_path = 'model_params/013/'
param_path = 'model_params/' + str(args.n).zfill(3) + "/"

with open(args.file, 'w+') as f:
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
