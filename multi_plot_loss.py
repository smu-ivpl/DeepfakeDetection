import argparse

import numpy as np

from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot losses from log")
    parser.add_argument("--log-file1", help="path to log file", default='./logs/deit_d_555_eye')
    parser.add_argument("--log-file2", help="path to log file", default='./logs/deit_d_555_eyeA')
    parser.add_argument("--fake-weight", help="weight for fake loss", default=1.5, type=float) #1.5 1 0.5
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.log_file1, "r") as f:
        lines = f.readlines()
    with open(args.log_file2, "r") as f:
        lines2 = f.readlines()
    #with open(args.log_file3, "r") as f:
    #    lines3 = f.readlines()
    #with open(args.log_file4, "r") as f:
    #    lines4 = f.readlines()
    real_losses = []
    fake_losses = []
    real_losses2 = []
    fake_losses2 = []
    '''
    real_losses3 = []
    fake_losses3 = []
    real_losses4 = []
    fake_losses4 = []
    '''
    for line in lines:
        line = line.strip()
        if line.startswith("fake_loss"):
            fake_losses.append(float(line.split(" ")[-1]))
        elif line.startswith("real_loss"):
            real_losses.append(float(line.split(" ")[-1]))
    for line in lines2:
        line = line.strip()
        if line.startswith("fake_loss"):
            fake_losses2.append(float(line.split(" ")[-1]))
        elif line.startswith("real_loss"):
            real_losses2.append(float(line.split(" ")[-1]))
    '''
    for line in lines3:
        line = line.strip()
        if line.startswith("fake_loss"):
            fake_losses3.append(float(line.split(" ")[-1]))
        elif line.startswith("real_loss"):
            real_losses3.append(float(line.split(" ")[-1]))
    for line in lines4:
        line = line.strip()
        if line.startswith("fake_loss"):
            fake_losses4.append(float(line.split(" ")[-1]))
        elif line.startswith("real_loss"):
            real_losses4.append(float(line.split(" ")[-1]))
    '''

    real_losses = np.array(real_losses)
    fake_losses = np.array(fake_losses)
    real_losses2 = np.array(real_losses2)
    fake_losses2 = np.array(fake_losses2)
    '''
    real_losses3 = np.array(real_losses3)
    fake_losses3 = np.array(fake_losses3)
    real_losses4 = np.array(real_losses4)
    fake_losses4 = np.array(fake_losses4)
    '''
    loss = (fake_losses * args.fake_weight + real_losses)/2
    loss2 = (fake_losses2 * args.fake_weight + real_losses2) / 2
    #loss3 = (fake_losses3 * args.fake_weight + real_losses3) / 2
    #loss4 = (fake_losses4 * args.fake_weight + real_losses4) / 2
    plt.title("Weighted loss ({}*fake_loss + real_loss)/2".format(args.fake_weight))
    # ignore early epochs  loss is quite noisy and there could be spikes
    '''
    for idx in best_loss_idx:
        plt.annotate(str(idx), (idx, loss[idx]))
    '''
    plt.xlabel('epoch')
    plt.ylabel('validation loss')
    plt.plot(loss, linestyle='--', linewidth=1)
    plt.plot(loss2, linestyle='-', linewidth=2)
    #plt.plot(loss3, linestyle=':', linewidth=2)
    #plt.plot(loss4, marker='+', color='y')
    plt.legend(['eye', 'eyeA'])
    plt.savefig("weight1.5.pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
