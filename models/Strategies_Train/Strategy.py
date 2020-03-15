from abc import ABC, abstractmethod

class Strategy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def applyStrategy(self, X_train, y_train, **kwargs):
        pass