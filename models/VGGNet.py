from . import Model
from keras.models import Sequential
import Data

class VGGNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(VGGNet, self).__init__(data, *args)

    def build(self, *args, trainedModel=None) -> Sequential:
        pass