from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from util.log_util import create_args_str


def visualize_cost(target_dir, args):
    stats_path = join(target_dir, 'stats.tsv')
    data = np.loadtxt(stats_path, delimiter='\t', skiprows=1)
    epochs = data[:, 0]
    ctc_train = data[:, 1]
    ctc_val = data[:, 2]
    ctc_train_mean = data[:, 3]
    ctc_val_mean = data[:, 4]
    ler_train = data[:, 5]
    ler_val = data[:, 6]
    ler_train_mean = data[:, 7]
    ler_val_mean = data[:, 8]

    fig_ctc = create_figure(ctc_train, ctc_train_mean, ctc_val, ctc_val_mean, create_title('CTC', epochs, args))
    fig_ler = create_figure(ler_train, ler_train_mean, ler_val, ler_val_mean, create_title('LER', epochs, args))

    return fig_ctc, fig_ler, int(max(epochs))


def create_title(loss_type, epochs, args):
    title = f'{loss_type} loss after {int(max(epochs))} epochs'
    parms = create_args_str(args, ['language', 'feature_type', 'synthesize'])
    return title + f' ({parms})'


def create_figure(loss_train, loss_train_mean, loss_val, loss_val_mean, title):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    t, = ax.plot(loss_train, label='training')
    t_mean, = ax.plot(loss_train_mean, linestyle='dashed', label='training (mean)')

    v, = ax.plot(loss_val, label='validation')
    v_mean, = ax.plot(loss_val_mean, linestyle='dashed', label='validation (mean)')

    ax.legend(handles=[t, t_mean, v, v_mean])
    return fig
