from . import Model, AlexNet, VGGNet
import config

class ModelFactory:

    def __init__(self):
        pass

    def getModel(self, modelType, *args, **kwargs) -> Model:

        try:

            if modelType == config.ALEX_NET:
                return AlexNet.AlexNet(*args, **kwargs)
            elif modelType == config.VGG_NET:
                return VGGNet.VGGNet(*args, **kwargs)
            else:
                return AttributeError()

        except:
            raise