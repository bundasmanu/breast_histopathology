from abc import ABC, abstractmethod
from keras.models import Sequential
from typing import Tuple, List
from .Strategies_Train import OverSampling, UnderSampling, DataAugmentation, Strategy
from keras.callbacks.callbacks import History
import config
from exceptions import CustomError
import numpy as np
import keras
import Data

class Model(ABC):

    StrategyList = list()

    @abstractmethod
    def __init__(self, numberCNNLayers, numberDenseLayers):
        self.nCNNLayers = numberCNNLayers
        self.nDenseLayers = numberDenseLayers

    @abstractmethod
    def define_train_strategies(self, undersampling=True, oversampling=False, data_augmentation=False) -> bool:

        '''
        THIS FUNCTION ESTABILISHES TRAINING STRATEGIES (UNDER SAMPLING AND OVER SAMPLING ARE INDEPENDENT, USER ONLY ACCEPTS ONE)
        :param underSampling: boolean --> True wants undersampling strategy
        :param oversampling: boolean --> True wants oversampling strategy
        :param dataAugmentation: boolean --> True wants data augmentation strategy
        :return: boolean --> True no errors occured, False --> problem on the definition of any strategy
        '''

        try:

            if undersampling == True and oversampling == True:
                raise CustomError.ErrorCreationModel(config.ERROR_INCOHERENT_STRATEGY)
                return False

            if undersampling == True:
                self.StrategyList.append(UnderSampling.UnderSampling())
            else:
                self.StrategyList.append(OverSampling.OverSampling())

            if data_augmentation == True:
                self.StrategyList.append(DataAugmentation.DataAugmentation())

            return True
        except:
            raise

    def template_method(self, data : Data.Data, *args) -> Tuple[Sequential, np.array, History]:

        '''
        https://refactoring.guru/design-patterns/template-method/python/example
        THIS FUNCTION REPRESENTS A TEMPLATE PATTERN TO EXECUTE THE ALL SEQUENCE OF JOBS TO DO
        :return: Sequential: trained model
        :return: numpy array: model test data predictions
        :return History.history: history of trained model
        '''

        try:
            model = self.build(*args)
            no_errors = self.define_train_strategies() #THIS FUNCTION IS APPLIED ON INHERITED OBJECTS OF THIS CLASS (ALEX_NET OR VGG NET)
            if no_errors == False:
                raise
            history, model = self.train(model, data)
            predictions = self.predict(model, data)

            return model, predictions, history
        except:
            raise CustomError.ErrorCreationModel(config.ERROR_MODEL_EXECUTION)

    @abstractmethod
    def build(self, *args, trainedModel=None) -> Sequential: #I PUT TRAINED MODEL ARGUMENT AFTER ARGS BECAUSE NON REQUIRED ARGUMENTS NEED TO BE AFTER *ARGS
        pass

    @abstractmethod
    def train(self, model : Sequential, data : Data.Data) -> Tuple[History, Sequential]:
        pass

    def predict(self, model : Sequential, data : Data.Data):

        try:

            predictions = model.predict(
                x=data.X_test,
                use_multiprocessing=config.MULTIPROCESSING
            )

            #CHECK PREDICTIONS OUTPUT WITH REAL TARGETS
            argmax_preds = np.argmax(predictions, axis=1) #BY ROW, BY EACH SAMPLE

            #I APPLY ONE HOT ENCODING, IN ORDER TO FACILITATE COMPARISON BETWEEN Y_TEST AND PREDICTIONS
            argmax_preds = keras.utils.to_categorical(argmax_preds)

            return argmax_preds

        except:
            raise

    @abstractmethod
    def __str__(self):
        return "Model(nº CNN : ", self.nCNNLayers, " nº Dense: ", self.nDenseLayers