from abc import ABC, abstractmethod

import librosa
import numpy as np

from util.audio_util import ms_to_frames


class Audible(ABC):
    # cache values
    _audio = None
    _rate = None
    _mag_specgram = None
    _pow_specgram = None
    _mel_specgram = None
    _mfcc = None

    @property
    def audio(self):
        if self._audio is not None:
            return self._audio
        self.audio, self._rate = self._create_audio_and_rate()
        return self._audio

    @audio.setter
    def audio(self, audio):
        self._audio = audio.astype(np.float32)
        self._mag_specgram = None
        self._pow_specgram = None
        self._mel_specgram = None
        self._mfcc = None

    @property
    def rate(self):
        if self._rate is not None:
            return self._rate
        self._audio, self._rate = self._create_audio_and_rate()
        return self._audio

    @abstractmethod
    def _create_audio_and_rate(self):
        return None, None

    def mag_specgram(self, window_size=20, step_size=10, unit='ms'):
        if self._mag_specgram is not None:
            return self._mag_specgram

        if unit == 'ms':
            window_size = ms_to_frames(window_size, self.rate)
            step_size = ms_to_frames(step_size, self.rate)

        D = librosa.stft(self.audio, n_fft=window_size, hop_length=step_size)
        self._mag_specgram, phase = librosa.magphase(D)

        return self._mag_specgram

    def pow_specgram(self, window_size=20, step_size=10, unit='ms'):
        if self._pow_specgram is not None:
            return self._pow_specgram

        mag = self.mag_specgram(window_size, step_size, unit)
        self._pow_specgram = librosa.amplitude_to_db(mag, ref=np.max)
        return self._pow_specgram

    def mel_specgram(self, window_size=20, step_size=10, unit='ms', n_mels=40):
        if self._mel_specgram is not None:
            return self._mel_specgram

        if unit == 'ms':
            window_size = ms_to_frames(window_size, self.rate)
            step_size = ms_to_frames(step_size, self.rate)

        spec = librosa.feature.melspectrogram(y=self.audio, sr=self.rate,
                                              n_fft=window_size, hop_length=step_size, n_mels=n_mels)
        self._mel_specgram = librosa.power_to_db(spec, ref=np.max)
        return self._mel_specgram

    def mfcc(self, window_size=20, step_size=10, unit='ms', n_mels=40):
        if self._mfcc is not None:
            return self._mfcc

        if unit == 'ms':
            window_size = ms_to_frames(window_size, self.rate)
            step_size = ms_to_frames(step_size, self.rate)

        self._mfcc = librosa.feature.mfcc(self.audio, self.rate, n_mfcc=13,
                                          n_fft=window_size, hop_length=step_size, n_mels=n_mels)
        return self._mfcc
