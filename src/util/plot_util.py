import os

import matplotlib.pyplot as plt
import numpy as np

from util.log_util import create_args_str


def create_title(loss_type, epochs, args):
    title = f'{loss_type} loss after {epochs} epochs'
    parms = create_args_str(args, ['language', 'feature_type', 'synthesize'])
    return title + f' ({parms})'


def visualize_cost(target_dir, epochs, args):
    stats_path = os.path.join(target_dir, 'stats.tsv')
    data = np.loadtxt(stats_path, delimiter='\t', skiprows=1)
    ctc_train = data[:, 1]
    ler_train = data[:, 2]
    ctc_val = data[:, 3]
    ler_val = data[:, 4]

    fig_ctc = create_figure(ctc_train, ctc_val, create_title('CTC', epochs, args))
    fig_ler = create_figure(ler_train, ler_val, create_title('LER', epochs, args))

    return fig_ctc, fig_ler


def create_figure(loss_train, loss_val, title):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    t, = ax.plot(loss_train, label='train-set')
    v, = ax.plot(loss_val, label='dev-set')
    ax.legend(handles=[t, v])
    return fig
