import gzip
import os
import pickle
from copy import copy
from os.path import join


def save_corpus(corpus_entries, target_root, gzip=False):
    corpus_file = join(target_root, 'corpus')
    if gzip:
        corpus_file += '.gz'
        with gzip.open(corpus_file, 'wb') as corpus:
            corpus.write(pickle.dumps(corpus_entries))
    else:
        with open(corpus_file, 'wb') as corpus:
            pickle.dump(corpus_entries, corpus)
    return corpus_file


def load_corpus(corpus_root):
    corpus_file = join(corpus_root, 'corpus')
    print(f'loading {corpus_file} ...')
    if corpus_file.endswith('.gz'):
        with gzip.open(corpus_file, 'rb') as corpus_f:
            corpus = pickle.loads(corpus_f.read())
    else:
        with open(corpus_file, 'rb') as corpus_f:
            corpus = pickle.load(corpus_f)
    print(f'...done! Loaded {corpus.name}: {len(corpus)} corpus entries')
    return corpus


def find_file_by_extension(directory, extension):
    return next(iter(filename for filename in os.listdir(directory) if filename.lower().endswith(extension.lower())),
                None)


def filter_corpus_entry_by_subset_prefix(corpus_entries, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [corpus_entry for corpus_entry in corpus_entries
            if corpus_entry.subset
            and any(corpus_entry.subset.startswith(prefix) for prefix in prefixes)]


def calculate_crop(segments):
    crop_start = min(segment.start_frame for segment in segments)
    crop_end = max(segment.end_frame for segment in segments)
    return crop_start, crop_end


def crop_segments(segments):
    cropped_segments = []
    crop_start, crop_end = calculate_crop(segments)
    for segment in segments:
        cropped_segment = copy(segment)
        cropped_segment.start_frame -= crop_start
        cropped_segment.end_frame -= crop_start
        cropped_segments.append(cropped_segment)
    return cropped_segments