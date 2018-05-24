import os

import matplotlib.pyplot as plt
import numpy as np


def plot_results(target_dir):
    stats_path = os.path.join(target_dir, 'stats.tsv')
    data = np.loadtxt(stats_path, delimiter='\t', skiprows=1)
    train_cost = data[:,1]
    train_ler = data[:,2]
    val_cost = data[:,3]
    val_ler = data[:,4]

    fig = plt.figure(figsize=(16, 9))

    ax_ctc = fig.add_subplot(211)
    ax_ctc.set_title('CTC Loss')
    ax_ctc.set_xlabel('Epoch')
    ax_ctc.set_ylabel('CTC Loss')
    ax_ctc.plot(train_cost)
    ax_ctc.plot(val_cost)

    ax_ler = fig.add_subplot(212)
    ax_ler.set_title('LER Loss')
    ax_ler.set_xlabel('Epochs')
    ax_ler.set_ylabel('LER Loss')
    ax_ler.plot(train_ler)
    ax_ler.plot(val_ler)

    plt.show()


if __name__ == '__main__':
    target_dir = r'E:\2018-05-24-14-40-16'
    plot_results(target_dir)
