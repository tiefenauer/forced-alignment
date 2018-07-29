"""
Compares the speech segments detected by WebRTC with the ones derived from the corpus metadata by measuring precision
and recall.
"""
import argparse
from operator import itemgetter
from os import remove
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import savefig
from tabulate import tabulate
from tqdm import tqdm

from util.corpus_util import get_corpus
from util.log_util import redirect_to_file, reset_redirect
from util.vad_util import webrtc_voice

parser = argparse.ArgumentParser(description="""Measure WebRTC-VAD performance by calculating precision/recall""")
parser.add_argument('-c', '--corpus', nargs='?', type=str, choices=['rl', 'ls'], default=None,
                    help='(optional) corpus to use (default: all)')
args = parser.parse_args()


def calc_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def calc_intersections(a, b):
    a = sorted(a, key=itemgetter(0))
    b = sorted(b, key=itemgetter(0))
    for start_a, end_a in a:
        x = set(range(start_a, end_a + 1))
        for start_b, end_b in ((s, e) for (s, e) in b if calc_overlap((s, e), (start_a, end_a))):
            y = range(start_b, end_b + 1)
            intersection = x.intersection(y)
            if intersection:
                yield min(intersection), max(intersection)


def calculate_boundaries(corpus_entry):
    start_frames = (seg.start_frame for seg in corpus_entry.speech_segments)
    end_frames = (seg.end_frame for seg in corpus_entry.speech_segments)
    return np.array(list(zip(start_frames, end_frames)))


def calculate_boundaries_webrtc(corpus_entry, aggressiveness=3):
    voiced_segments = webrtc_voice(corpus_entry.audio, corpus_entry.rate, aggressiveness=aggressiveness)
    start_frames = (seg.start_frame for seg in voiced_segments)
    end_frames = (seg.end_frame for seg in voiced_segments)
    return np.array(list(zip(start_frames, end_frames)))


def precision_recall(corpus_entry, aggressiveness):
    boundaries_original = calculate_boundaries(corpus_entry)
    boundaries_webrtc = calculate_boundaries_webrtc(corpus_entry, aggressiveness=aggressiveness)

    intersections = calc_intersections(boundaries_original, boundaries_webrtc)
    n_frames_intersection = sum(len(range(start, end + 1)) for start, end in intersections)
    n_frames_original = sum(len(range(start, end + 1)) for start, end in boundaries_original)
    n_frames_webrtc = sum(len(range(start, end + 1)) for start, end in boundaries_webrtc)

    p = n_frames_intersection / (n_frames_webrtc + 1e-3)  # precision
    r = n_frames_intersection / (n_frames_original + 1e-3)  # recall
    f = 2.0 * p * r / (p + r + 1e-3)  # f-score

    return p, r, f, len(boundaries_original), len(boundaries_webrtc)


def compare_corpus(corpus, aggressiveness):
    p_r_f_d = []
    tot_orig, tot_webrtc = 0, 0
    progress = tqdm(enumerate(corpus, 1), total=len(corpus), unit='entries')
    for i, corpus_entry in progress:
        p, r, f, n_orig, n_webrtc = precision_recall(corpus_entry, aggressiveness)
        tot_orig += n_orig
        tot_webrtc += n_webrtc
        d = n_webrtc - n_orig  # difference in number of speech segments
        description = f'#original={n_orig} (avg={tot_orig / i:.3f}), ' \
                      f'#WebRTC={n_webrtc} (avg={tot_webrtc / i:.3f}), ' \
                      f'precision={p:.3f}, recall={r:.3f}, f-score={f:.3f}'
        progress.set_description(description)
        p_r_f_d.append((n_orig, n_webrtc, p, r, f, d))
        del corpus_entry._audio

    p_r_f_d = np.asarray(p_r_f_d)
    avg_orig, avg_webrtc, avg_p, avg_r, avg_f, avg_d = np.abs(p_r_f_d).mean(axis=0)
    ds = p_r_f_d[:, -1]
    avg_d_neg = np.extract(ds < 0, ds).mean()
    avg_d_pos = np.extract(ds > 0, ds).mean()

    return avg_orig, avg_webrtc, avg_p, avg_r, avg_f, avg_d, avg_d_neg, avg_d_pos


def create_corpus_stats(corpus):
    print(f'Comparing automatic/manual VAD for {corpus.name} corpus')
    stats = {'Aggressiveness': [0, 1, 2, 3],
             'Avg. # of speech segments (manual)': [],
             'Avg. # of speech segments (WebRTC)': [],
             'Precision': [],
             'Recall': [],
             'F-Score': [],
             'Difference': [],
             'Difference (negative)': [],
             'Difference (positive)': []}
    for aggressiveness in stats['Aggressiveness']:
        print(f'precision/recall with aggressiveness={aggressiveness}\n')
        avg_orig, avg_webrtc, avg_p, avg_r, avg_f, avg_d, avg_d_neg, avg_d_pos = compare_corpus(corpus, aggressiveness)
        stats['Avg. # of speech segments (manual)'].append(avg_orig)
        stats['Avg. # of speech segments (WebRTC)'].append(avg_webrtc)
        stats['Precision'].append(avg_p)
        stats['Recall'].append(avg_r)
        stats['F-Score'].append(avg_f)
        stats['Difference'].append(avg_d)
        stats['Difference (negative)'].append(avg_d_neg)
        stats['Difference (positive)'].append(avg_d_pos)

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
    ax2.set_ylabel('# speech segments')
    d_abs, = ax2.plot(x, np.array(stats['Difference']), color='c', label='Avg. Difference')
    d_neg, = ax2.plot(x, np.array(stats['Difference (negative)']), color='m', label='Avg. Difference (negative)')
    d_pos, = ax2.plot(x, np.array(stats['Difference (positive)']), color='y', label='Avg. Difference (positive)')

    lgd = plt.legend(handles=[p, r, f, d_abs, d_neg, d_pos], bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig, ax1, ax2, lgd


def save_stats(stats, corpus):
    stats_file = join(corpus.root_path, 'corpus.webrtc.stats')

    # save plot
    title = f'Comparison of automatic/manual VAD for {corpus.name} corpus ({len(corpus)} entries)'
    fig, ax1, ax2, lgd = plot_stats(stats, title)
    stats_plot_file = stats_file + '.png'
    if exists(stats_plot_file):
        remove(stats_plot_file)
    savefig(stats_plot_file, bbox_extra_artists=(lgd,), bbox_inches='tight')

    # save tabular stats
    print(f'Writing results to {stats_file}')
    redirect_to_file(stats_file)
    print(tabulate(stats, headers='keys'))
    reset_redirect()
    return stats


if __name__ == '__main__':
    corpora = [args.corpus] if args.corpus else ['rl', 'ls']
    for corpus_id in corpora:
        corpus = get_corpus(corpus_id)
        stats = create_corpus_stats(corpus)
        save_stats(stats, corpus)
