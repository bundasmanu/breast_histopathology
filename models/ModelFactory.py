from . import Model, AlexNet, VGGNet
import config

class ModelFactory:

    def __init__(self):
        pass

    def getModel(self, modelType, *args) -> Model:

        try:

            if modelType == config.ALEX_NET:
                return AlexNet.AlexNet(*args)
            elif modelType == config.VGG_NET:
                return VGGNet.VGGNet(*args)
            else:
                return AttributeError()

        except:
            raise