from . import Model
from keras.models import Sequential

class VGGNet(Model.Model):

    def build(self, trainedModel=None) -> Sequential:
        pass