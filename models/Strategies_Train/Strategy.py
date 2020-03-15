from abc import ABC, abstractmethod

class Strategy(ABC):

    @abstractmethod
    def applyStrategy(self):
        pass