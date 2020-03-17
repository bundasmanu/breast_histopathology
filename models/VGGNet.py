from . import Model
from keras.models import Sequential

class VGGNet(Model.Model):

    def build(self, *args, trainedModel=None) -> Sequential:
        pass