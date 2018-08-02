from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta

from tabulate import tabulate
from tqdm import tqdm

from util.corpus_util import filter_corpus_entry_by_subset_prefix


class Corpus(ABC):
    """
    Base class for corpora
    """

    def __init__(self, name, corpus_entries, root_path):
        """
        Create a new corpus holding a list of corpus entries
        :param name: unique corpus name
        :param corpus_entries: list of CorpusEntry instances
        :param root_path: path to directory holding the audio files for the corpus entries
        """
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
            print(f'filtering out speech segments with numbers in transcript')
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
        return sorted(set(lang for lang in (corpus_entry.language for corpus_entry in self.corpus_entries)))

    @property
    def keys(self):
        return sorted([corpus_entry.id for corpus_entry in self.corpus_entries])

    @abstractmethod
    def train_dev_test_split(self):
        """return training-, validation- and test-set
        Since these sets are constructed
        """
        pass

    def summary(self):
        print('')
        print(f'Corpus: {self.name}')
        print(self.root_path)
        table = {}
        t_entries = t_speeches = len_speeches = t_speeches_u = len_speeches_u = t_pauses = len_pauses = len_segments = t_segments = t_duration = 0
        for lang in self.languages:
            entries = [entry for entry in self.corpus_entries if entry.language == lang]
            n_entries = len(entries)
            t_entries += n_entries

            speech_segments = [segment for entry in entries for segment in entry.speech_segments]
            n_speeches = len(speech_segments)
            l_speeches = sum(segment.audio_length for segment in speech_segments)
            t_speeches += n_speeches
            len_speeches += l_speeches

            speeches_unaligned = [segment for entry in entries for segment in entry.speech_segments_unaligned]
            n_speeches_u = len(speeches_unaligned)
            l_speeches_u = sum(segment.audio_length for segment in speeches_unaligned)
            t_speeches_u += n_speeches_u
            len_speeches_u += l_speeches_u

            pauses = [ps for entry in entries for ps in entry.pause_segments]
            n_pauses = len(pauses)
            l_pauses = sum(segment.audio_length for segment in pauses)
            t_pauses += n_pauses
            len_pauses += l_pauses

            segments = [sg for entry in entries for sg in entry.segments]
            n_segments = len(segments)
            l_segments = sum([segment.audio_length for segment in segments])
            t_segments += n_segments
            len_segments += l_segments

            duration = sum(entry.audio_length for entry in self.corpus_entries if entry.language == lang)
            t_duration += duration

            table[lang] = (n_entries,
                           n_speeches, timedelta(seconds=int(l_speeches)),
                           n_speeches_u, timedelta(seconds=int(len_speeches_u)),
                           n_pauses, timedelta(seconds=int(l_pauses)),
                           n_segments, timedelta(seconds=int(l_segments)),
                           timedelta(seconds=int(duration)))

        table['total'] = (t_entries,
                          t_speeches, timedelta(seconds=int(len_speeches)),
                          t_speeches_u, timedelta(seconds=int(len_speeches_u)),
                          t_pauses, timedelta(seconds=int(len_speeches_u)),
                          t_segments, timedelta(seconds=int(len_segments)),
                          timedelta(seconds=int(t_duration))
                          )
        headers = ['lang', '#entries',
                   '#speeches', '',
                   '#speeches', '(unal.)',
                   '#pauses', '',
                   '#total', '',
                   'audio']
        print(tabulate([(k,) + v for k, v in table.items()], headers=headers))


class ReadyLinguaCorpus(Corpus):

    def __init__(self, corpus_entries, root_path):
        super().__init__('ReadyLingua', corpus_entries, root_path)

    def train_dev_test_split(self):
        speech_segments = [seg for corpus_entry in self.corpus_entries
                           for seg in corpus_entry.speech_segments_not_numeric]

        # 80/10/10 split
        n_entries = len(speech_segments)
        train_split = int(n_entries * 0.8)
        test_split = int(train_split + (n_entries - train_split) / 2)

        train_set = speech_segments[:train_split]
        dev_set = speech_segments[train_split:test_split]
        test_set = speech_segments[test_split:]
        return train_set, dev_set, test_set


class LibriSpeechCorpus(Corpus):

    def __init__(self, corpus_entries, root_path):
        super().__init__('LibriSpeech', corpus_entries, root_path)

    def train_dev_test_split(self):
        train_entries = filter_corpus_entry_by_subset_prefix(self.corpus_entries, 'train-')
        dev_entries = filter_corpus_entry_by_subset_prefix(self.corpus_entries, 'dev-')
        test_entries = filter_corpus_entry_by_subset_prefix(self.corpus_entries, ['test-', 'unknown'])
        train_set = [seg for corpus_entry in train_entries for seg in corpus_entry.speech_segments_not_numeric]
        dev_set = [seg for corpus_entry in dev_entries for seg in corpus_entry.speech_segments_not_numeric]
        test_set = [seg for corpus_entry in test_entries for seg in corpus_entry.speech_segments_not_numeric]
        return train_set, dev_set, test_set
