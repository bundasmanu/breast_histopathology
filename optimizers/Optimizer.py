from abc import ABC, abstractmethod
from models import Model
from exceptions import CustomError
import config
import Data

class Optimizer(ABC):

    def __init__(self, model : Model.Model, individuals, iterations, dimensions):
        if model == None:
            return CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)
        self.model = model
        self.indiv = individuals
        self.iters = iterations
        self.dims = dimensions

    @abstractmethod
    def objectiveFunction(self, score, *args):

        '''

        :param score: final score on train
        :param args: values of model layers --> the number of args need to be equal to total of layer used
                    e.g: args: (32, 48, 64, 32, 16) --> the sum of nCNNLayers and nDenseLayers need to be equal to number of args
        :return: lost function
        '''

        try:

            cnnFilters = [args[i] for i in range(self.model.nCNNLayers)]
            totalFilters = sum(cnnFilters)
            denseNeurons = [args[(self.model.nCNNLayers+self.model.nDenseLayers) - i] for i in range(self.model.nDenseLayers)]
            totalNeurons = sum(denseNeurons)
            return 1.5 * ((1.0 - (1.0 / (totalFilters)))
                      + (1.0 - (1.0 / (totalNeurons)))) + 5.0 * (1.0 - score)

        except:
            raise

    @abstractmethod
    def optimize(self, data : Data.Data):
        pass