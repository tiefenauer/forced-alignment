from abc import ABC, abstractmethod

from util.audio_util import pow_specgram


class Audible(ABC):
    # cache values
    _audio = None
    _rate = None

    @property
    def audio(self):
        if self._audio is not None:
            return self._audio
        self._audio, self._rate = self._create_audio_and_rate()
        return self._audio

    @audio.setter
    def audio(self, audio):
        self._audio = audio

    @property
    def rate(self):
        if self._rate is not None:
            return self._rate
        self._audio, self._rate = self._create_audio_and_rate()
        return self._audio

    @abstractmethod
    def _create_audio_and_rate(self):
        return None, None

    def spectrogram(self, window_size=20, step_size=10, unit='ms'):
        return pow_specgram(self.audio, self.rate, window_size=window_size, step_size=step_size, unit=unit)
