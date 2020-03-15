from models import Model
import config
from exceptions import CustomError
from . import PSO, GA

class OptimizerFactory:

    def __init__(self):
        pass

    def createOptimizer(self, typeOptimizer : str, model : Model):

        try:

            if model is None:
                raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)

            if typeOptimizer == config.PSO_OPTIMIZER:
                return PSO(model)
            elif typeOptimizer == config.GA_OPTIMIZER:
                return GA(model)
            else:
                raise CustomError.ErrorCreationModel(config.ERROR_INVALID_OPTIMIZER)

        except:
            raise