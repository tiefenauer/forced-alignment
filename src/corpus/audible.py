from abc import ABC, abstractmethod


class Audible(ABC):

    @property
    @abstractmethod
    def audio(self):
        return None

    @property
    @abstractmethod
    def rate(self):
        return None
