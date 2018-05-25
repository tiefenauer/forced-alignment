from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta

from tabulate import tabulate
from tqdm import tqdm

from util.corpus_util import filter_corpus_entry_by_subset_prefix


class Corpus(ABC):

    def __init__(self, name, corpus_entries, root_path):
        self._name = ''
        self.name = name
        for corpus_entry in corpus_entries:
            corpus_entry.corpus = self
        self.corpus_entries = corpus_entries
        self.root_path = root_path

    def __iter__(self):
        for corpus_entry in self.corpus_entries:
            yield corpus_entry

    def __getitem__(self, val):
        # access by index
        if isinstance(val, int) or isinstance(val, slice):
            return self.corpus_entries[val]
        # access by id
        if isinstance(val, str):
            return next(iter([corpus_entry for corpus_entry in self.corpus_entries if corpus_entry.id == val]), None)
        return None

    def __len__(self):
        return len(self.corpus_entries)

    def __call__(self, *args, **kwargs):
        languages = kwargs['languages'] if 'languages' in kwargs else self.languages
        include_numeric = kwargs['include_numeric'] if 'include_numeric' in kwargs else True
        print(f'filtering languages={languages}')
        entries = [entry for entry in self.corpus_entries if entry.language in languages]
        print(f'found {len(entries)} entries for languages {languages}')

        if not include_numeric:
            print(f'filtering out speech segments with numbers in transcription')
            entries = [entry(include_numeric=include_numeric) for entry in tqdm(entries, unit=' entries')]

        _copy = deepcopy(self)
        _copy.corpus_entries = entries
        return _copy

    @property
    def name(self):
        languages = ', '.join(self.languages)
        return self._name + f' (languages: {languages})'

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def languages(self):
        return set(lang for lang in (corpus_entry.language for corpus_entry in self.corpus_entries))

    @property
    def keys(self):
        return [corpus_entry.id for corpus_entry in self.corpus_entries]

    @abstractmethod
    def train_dev_test_split(self):
        """return training-, validation- and test-set
        Since these sets are constructed
        """
        pass

    def summary(self):
        print('')
        print(f'Corpus: {self.name}')
        table = {}
        total_entries = total_sg = total_sp = total_su = total_ps = total_length = 0
        for lang in self.languages:
            entries = [entry for entry in self.corpus_entries if entry.language == lang]
            n_entries = len(entries)
            n_segments = len([sg for entry in entries for sg in entry.segments])
            n_speeches = len([sp for entry in entries for sp in entry.speech_segments])
            n_speeches_u = len([su for entry in entries for su in entry.speech_segments_unaligned])
            n_pauses = len([ps for entry in entries for ps in entry.pause_segments])
            length = sum(entry.audio_length for entry in self.corpus_entries if entry.language == lang)

            total_entries += n_entries
            total_sg += n_segments
            total_sp += n_speeches
            total_su += n_speeches_u
            total_ps += n_pauses
            total_length += length

            table[lang] = (n_entries, n_segments, n_speeches, n_speeches_u, n_pauses, timedelta(seconds=int(length)))
        table['total'] = (total_entries, total_sg, total_sp, total_su, total_ps, timedelta(seconds=total_length))
        print(tabulate([(k,) + v for k, v in table.items()],
                       headers=['language', '# entries', '# segments', '# speeches', '# speeches (unaligned)', '# pauses', 'hh:mm:ss']))


class ReadyLinguaCorpus(Corpus):

    def __init__(self, corpus_entries, root_path):
        super().__init__('ReadyLingua', corpus_entries, root_path)

    def train_dev_test_split(self):
        n_entries = len(self.corpus_entries)
        # 80/10/10 split
        train_split = int(n_entries * 0.8)
        test_split = int(train_split + (n_entries - train_split) / 2)

        train_set = self.corpus_entries[:train_split]
        dev_set = self.corpus_entries[train_split:test_split]
        test_set = self.corpus_entries[test_split:]
        return train_set, dev_set, test_set


class LibriSpeechCorpus(Corpus):

    def __init__(self, corpus_entries, root_path):
        super().__init__('LibriSpeech', corpus_entries, root_path)

    def train_dev_test_split(self):
        train_set = filter_corpus_entry_by_subset_prefix(self.corpus_entries, 'train-')
        dev_set = filter_corpus_entry_by_subset_prefix(self.corpus_entries, 'dev-')
        test_set = filter_corpus_entry_by_subset_prefix(self.corpus_entries, ['test-', 'unknown'])
        return train_set, dev_set, test_set
