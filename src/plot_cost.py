import argparse
import os
from pathlib import Path

from util.plot_util import visualize_cost

parser = argparse.ArgumentParser(description="""Plot CTC and LER cost from previous train run""")
parser.add_argument('target_dir', type=str, nargs='?', help='(optional) directory to read from')
args = parser.parse_args()


def show_plot(target_dir):
    epochs = len(
        Path(os.path.join(target_dir, 'stats.tsv')).read_text().split('\n')) - 2  # header line and empty last line
    fig_ctc, fig_ler = visualize_cost(target_dir, epochs)
    fig_ctc.show()
    fig_ler.show()


if __name__ == '__main__':
    target_dir = r'E:\2018-07-11-11-45-16_poc_1_mfcc'
    if args.target_dir:
        target_dir = args.target_dir

    show_plot(target_dir)
