import os
from operator import itemgetter
from os import remove

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import savefig
from os.path import exists
from tabulate import tabulate
from tqdm import tqdm

from definitions import CORPUS_TARGET_ROOT
from util.corpus_util import load_corpus
from util.log_util import print_to_file_and_console
from util.webrtc_util import split_segments

rl_corpus_root = os.path.join(CORPUS_TARGET_ROOT, 'readylingua-corpus')
ls_corpus_root = os.path.join(CORPUS_TARGET_ROOT, 'librispeech-corpus')


def calc_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def calc_intersection(a, b):
    a = sorted(a, key=itemgetter(0))
    b = sorted(b, key=itemgetter(0))
    for start_a, end_a in a:
        x = set(range(start_a, end_a + 1))
        for start_b, end_b in ((s, e) for (s, e) in b if calc_overlap((s, e), (start_a, end_a))):
            y = range(start_b, end_b + 1)
            intersection = x.intersection(y)
            if intersection:
                yield min(intersection), max(intersection)


def calculate_boundaries(segments):
    start_frames = (seg.start_frame for seg in segments)
    end_frames = (seg.end_frame for seg in segments)
    return np.array(list(zip(start_frames, end_frames)))


def calculate_boundaries_webrtc(corpus_entry, aggressiveness=3):
    voiced_segments, _ = split_segments(corpus_entry, aggressiveness=aggressiveness)
    boundaries = []
    for frames in voiced_segments:
        start_time = frames[0].timestamp
        end_time = (frames[-1].timestamp + frames[-1].duration)
        boundaries.append((start_time, end_time))
    return 2 * np.array(boundaries), voiced_segments


def precision_recall(corpus_entry, aggressiveness):
    boundaries_original = calculate_boundaries(corpus_entry.speech_segments)
    boundaries_webrtc, _ = calculate_boundaries_webrtc(corpus_entry, aggressiveness=aggressiveness)
    boundaries_webrtc = boundaries_webrtc * corpus_entry.rate  # convert to frames
    boundaries_webrtc = boundaries_webrtc.astype(int)

    intersections = calc_intersection(boundaries_original, boundaries_webrtc)
    n_frames_intersection = sum(len(range(start, end + 1)) for start, end in intersections)
    n_frames_original = sum(len(range(start, end + 1)) for start, end in boundaries_original)
    n_frames_webrtc = sum(len(range(start, end + 1)) for start, end in boundaries_webrtc)

    p = n_frames_intersection / (n_frames_webrtc + 1e-3)
    r = n_frames_intersection / (n_frames_original + 1e-3)
    f = 2.0 * p * r / (p + r + 1e-3)
    d = len(boundaries_webrtc) - len(boundaries_original)

    return p, r, f, d


def compare_corpus(corpus, aggressiveness):
    p_r_f_d = list(
        tqdm((precision_recall(corpus_entry, aggressiveness) for corpus_entry in corpus), total=len(corpus)))
    p_r_f_d = np.asarray(p_r_f_d)
    avg_p, avg_r, avg_f, avg_d = np.abs(p_r_f_d).mean(axis=0)
    ds = p_r_f_d[:, 3]
    avg_d_neg = np.extract(ds < 0, ds).mean()
    avg_d_pos = np.extract(ds > 0, ds).mean()

    return avg_p, avg_r, avg_f, avg_d, avg_d_neg, avg_d_pos


def create_corpus_stats(corpus):
    print(f'Comparing automatic/manual VAD for {corpus.name} corpus')
    stats = {'Aggressiveness': [0, 1, 2, 3], 'Precision': [], 'Recall': [], 'F-Score': [], 'Difference': [],
             'Difference (negative)': [], 'Difference (positive)': []}
    for aggressiveness in stats['Aggressiveness']:
        print(f'precision/recall with aggressiveness={aggressiveness}\n')
        avg_p, avg_r, avg_f, avg_d, avg_d_neg, avg_d_pos = compare_corpus(corpus, aggressiveness)
        stats['Precision'].append(avg_p)
        stats['Recall'].append(avg_r)
        stats['F-Score'].append(avg_f)
        stats['Difference'].append(avg_d)
        stats['Difference (negative)'].append(avg_d_neg)
        stats['Difference (positive)'].append(avg_d_pos)

    # save tabular stats
    stats_file = os.path.join(corpus.root_path, 'corpus.webrtc.stats')
    if exists(stats_file):
        remove(stats_file)
    print(f'Writing results to {stats_file}')
    f = print_to_file_and_console(stats_file)
    print(tabulate(stats, headers='keys'))
    f.close()

    # save plot
    fig, ax1, ax2, lgd = plot_stats(stats, f'Comparison of automatic/manual VAD for {corpus.name} corpus')
    stats_plot_file = stats_file + '.png'
    if exists(stats_plot_file):
        remove(stats_plot_file)
    savefig(stats_plot_file, bbox_extra_artists=(lgd,), bbox_inches='tight')
    return stats


def plot_stats(stats, title=None):
    x = stats['Aggressiveness']

    fig, ax1 = plt.subplots(figsize=(12, 5), facecolor='white')
    if title:
        ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xlabel('aggressiveness')
    ax1.set_ylabel('precision/recall/F-score')
    p, = ax1.plot(x, np.array(stats['Precision']), color='r', label='Precision')
    r, = ax1.plot(x, np.array(stats['Recall']), color='g', label='Recall')
    f, = ax1.plot(x, np.array(stats['F-Score']), color='b', label='F-Score')

    ax2 = ax1.twinx()
    ax2.set_ylabel('difference')
    d_abs, = ax2.plot(x, np.array(stats['Difference']), color='c', label='Difference')
    d_neg, = ax2.plot(x, np.array(stats['Difference (negative)']), color='m', label='Difference (negative)')
    d_pos, = ax2.plot(x, np.array(stats['Difference (positive)']), color='y', label='Difference (positive)')

    lgd = plt.legend(handles=[p, r, f, d_abs, d_neg, d_pos], bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig, ax1, ax2, lgd


if __name__ == '__main__':
    rl_corpus = load_corpus(rl_corpus_root)
    create_corpus_stats(rl_corpus)
    ls_corpus = load_corpus(ls_corpus_root)
    create_corpus_stats(ls_corpus)
