from itertools import chain, repeat


class DummyCorpus(object):
    """Helper class to repeat a given list of corpus entries a certain number of times. Additionally, the number of
     speech segments on each entry can be limited. An instance of this class is an iterator that will iterate over all
     corpus entries N times (i.e. each corpus entry will be yielded N times).

     This is useful for example for a POC where a RNN learns from only the first 5 speech segments of a single instance.
     """

    def __init__(self, repeat_samples, times, num_segments=None):
        self.repeat_samples = repeat_samples
        self.times = times
        self.num_segments = num_segments

    def __iter__(self):
        for repeat_sample in chain.from_iterable(repeat(self.repeat_samples, self.times)):
            segments_with_text = [speech for speech in repeat_sample.speech_segments_not_numeric
                                  if speech.text and len(speech.audio) > 0]
            if self.num_segments:
                segments_with_text = segments_with_text[:self.num_segments]
            repeat_sample.segments = segments_with_text
            yield repeat_sample

    def __len__(self):
        return self.times * len(self.repeat_samples)