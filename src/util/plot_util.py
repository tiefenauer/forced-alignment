import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visualize_cost(target_dir, epochs):
    stats_path = os.path.join(target_dir, 'stats.tsv')
    data = np.loadtxt(stats_path, delimiter='\t', skiprows=1)
    ctc_train = data[:, 1]
    ler_train = data[:, 2]
    ctc_val = data[:, 3]
    ler_val = data[:, 4]

    fig_ctc = create_figure(ctc_train, ctc_val, f'CTC loss (convergence after {epochs} epochs)')
    fig_ler = create_figure(ler_train, ler_val, f'LER loss (convergence after {epochs} epochs)')

    return fig_ctc, fig_ler


def create_figure(loss_train, loss_val, title):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    t, = ax.plot(loss_train, label='train-set')
    v, = ax.plot(loss_val, label='dev-set')
    ax.legend(handles=[t,v])
    return fig


def show_plot(target_dir):
    epochs = len(Path(os.path.join(target_dir, 'stats.tsv')).read_text().split('\n')) - 2 # header line and empty last line
    fig_ctc, fig_ler = visualize_cost(target_dir, epochs)
    fig_ctc.show()
    fig_ler.show()


if __name__ == '__main__':
    target_dir = r'E:\2018-07-11-09-54-24_poc_1_mfcc'
    show_plot(target_dir)
