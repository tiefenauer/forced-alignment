import argparse
import os

from util.plot_util import visualize_cost

parser = argparse.ArgumentParser(description="""Plot CTC and LER cost from previous train run""")
parser.add_argument('target_dir', type=str, nargs='?', help='(optional) directory to read from')
args = parser.parse_args()


def parse_args_line(line):
    _args = argparse.Namespace()
    for key, value in [tuple(arg.split('=')) for arg in line.strip().split(', ')]:
        setattr(_args, key, value)
    return _args


def show_plot(target_dir):
    _args = parse_args_line(open(os.path.join(target_dir, 'train.log')).readline())
    fig_ctc, fig_ler, epochs = visualize_cost(target_dir, _args)
    fig_ctc.show()
    fig_ctc.savefig(os.path.join(target_dir, f'ctc_cost_{epochs}_epochs.png'), bbox_inches='tight')
    fig_ler.show()
    fig_ler.savefig(os.path.join(target_dir, f'ler_cost_{epochs}_epochs.png'), bbox_inches='tight')


if __name__ == '__main__':
    target_dir = r'E:\2018-07-12-08-26-44_poc_5_en_mfcc_original'
    if args.target_dir:
        target_dir = args.target_dir

    show_plot(target_dir)
