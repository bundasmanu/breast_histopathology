from abc import ABC, abstractmethod
from keras.models import Sequential

class Model(ABC):

    @abstractmethod
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        super.__init__()

    def template_method(self):
        pass

    @abstractmethod
    def build_model(self, trainedModel=None) -> Sequential:
        pass

    @abstractmethod
    def train(self, model : Sequential) -> Sequential:
        pass

    @abstractmethod
    def predict(self, model : Sequential):
        pass