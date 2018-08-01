import argparse
from os.path import join

from util.plot_util import visualize_cost

parser = argparse.ArgumentParser(description="""Plot CTC and LER cost from previous train run""")
parser.add_argument('target_dir', type=str, nargs='?', help='(optional) directory to read from')
args = parser.parse_args()


def parse_args_line(line):
    _args = argparse.Namespace()
    prefix = '[YYYY-MM-DD hh:mm:ss - INFO ]'
    for key, value in [tuple(arg.split('=')) for arg in line[len(prefix):].strip().split(', ')]:
        setattr(_args, key, value)
    return _args


def show_plot(target_dir):
    _args = parse_args_line(open(join(target_dir, 'train.log')).readline())
    fig_ctc, fig_ler, epochs = visualize_cost(target_dir, _args)

    fig_ctc.show()
    fig_ctc_path = join(target_dir, f'ctc_cost_{epochs}_epochs.png')
    fig_ctc.savefig(fig_ctc_path, bbox_inches='tight')
    print(f'CTC plot saved to {fig_ctc_path}')

    fig_ler.show()
    fig_ler_path = join(target_dir, f'ler_cost_{epochs}_epochs.png')
    fig_ler.savefig(fig_ler_path, bbox_inches='tight')
    print(f'CTC plot saved to {fig_ler_path}')


if __name__ == '__main__':
    target_dir = r'E:\2018-08-01-13-56-48_RNN_poc1_rl_de_mfcc'
    if args.target_dir:
        target_dir = args.target_dir

    show_plot(target_dir)
