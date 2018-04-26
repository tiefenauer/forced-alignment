import gzip
import os
import pickle


def save_corpus(corpus_entries, corpus_file, gzip=False):
    if gzip:
        corpus_file += '.gz'
        with gzip.open(corpus_file, 'wb') as corpus:
            corpus.write(pickle.dumps(corpus_entries))
    else:
        with open(corpus_file, 'wb') as corpus:
            pickle.dump(corpus_entries, corpus)


def load_corpus(corpus_file):
    print(f'loading {corpus_file} ...')
    if corpus_file.endswith('.gz'):
        with gzip.open(corpus_file, 'rb') as corpus:
            corpus_entries = pickle.loads(corpus.read())
    else:
        with open(corpus_file, 'rb') as corpus:
            corpus_entries = pickle.load(corpus)
    print(f'...done! Loaded {corpus_entries.name}: {len(corpus_entries)} corpus entries')
    return corpus_entries


def find_file_by_extension(directory, extension):
    return next(iter(filename for filename in os.listdir(directory) if filename.lower().endswith(extension.lower())),
                None)


def filter_corpus_entry_by_subset_prefix(corpus_entries, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [corpus_entry for corpus_entry in corpus_entries
            if corpus_entry.subset
            and any(corpus_entry.subset.startswith(prefix) for prefix in prefixes)]