from abc import ABC, abstractmethod

from util.audio_util import pow_specgram


class Audible(ABC):

    @property
    @abstractmethod
    def audio(self):
        return None

    @property
    @abstractmethod
    def rate(self):
        return None

    def spectrogram(self, window_size=20, step_size=10, unit='ms'):
        return pow_specgram(self.audio, self.rate, window_size=window_size, step_size=step_size, unit=unit)
