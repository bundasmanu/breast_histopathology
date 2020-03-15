from . import Model, AlexNet, VGGNet
import config

class ModelFactory:

    def __init__(self) -> None:
        pass

    def getModel(self, modelType : str) -> Model:

        try:

            if modelType == config.ALEX_NET:
                return AlexNet()
            elif modelType == config.VGG_NET:
                return VGGNet()
            else:
                return AttributeError()

        except:
            raise