from abc import ABC, abstractmethod
from models import Model
from exceptions import CustomError
import config

class Optimizer(ABC):

    def __init__(self, model : Model.Model, individuals, iterations, dimensions):
        if model == None:
            return CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)
        self.model = model
        self.indiv = individuals
        self.iters = iterations
        self.dims = dimensions

    @abstractmethod
    def objectiveFunction(self, acc, *args):

        '''

        :param score: final score on train
        :param args: values of model layers --> the number of args need to be equal to total of layer used
                    e.g: args: (32, 48, 64, 32, 16) --> the sum of nCNNLayers and nDenseLayers need to be equal to number of args
                    last argument is a confusion matrix
        :return: lost function
        '''

        try:

            cnnFilters = [args[i]*i for i in range(self.model.nCNNLayers)] #ATTRIBUTION IMPORTANCE TO CNN FILTERS (*i) --> LAST FCONVOLUTION LAYER IS MORE IMPORTANT THAN FIRST
            totalFilters = sum(cnnFilters)
            denseNeurons = [args[(self.model.nCNNLayers+self.model.nDenseLayers) - (i+1)] for i in range(self.model.nDenseLayers)]
            totalNeurons = sum(denseNeurons)

            confusion_mat = args[-1]
            recall_idc = confusion_mat[1][0] #THIS TWO VALUES NEED TO BE OPTIMIZED TO BE THE MINIMUM
            precision_idc = confusion_mat[0][1]

            return 2.0 * ((1.0 - (1.0 / (totalFilters)))
                      + (1.0 - (1.0 / (totalNeurons)))) + 1.5 * (1.0 - acc) + 5.0 * (1.0 - recall_idc) + 3.0 * (1.0 - precision_idc)

        except:
            raise

    @abstractmethod
    def optimize(self):
        pass