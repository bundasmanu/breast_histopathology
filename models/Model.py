from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.callbacks.callbacks import History
from typing import Tuple, List
from .Strategies_Train import OverSampling, UnderSampling, DataAugmentation, Strategy
from keras.optimizers import Adam
from keras.callbacks.callbacks import History
import config
from exceptions import CustomError
import numpy as np

class Model(ABC):

    StrategyList = list()

    @abstractmethod
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

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

    def template_method(self) -> Tuple[Sequential, np.array, History]:

        '''
        https://refactoring.guru/design-patterns/template-method/python/example
        THIS FUNCTION REPRESENTS A TEMPLATE PATTERN TO EXECUTE THE ALL SEQUENCE OF JOBS TO DO
        :return: Sequential: trained model
        :return: numpy array: model test data predictions
        :return History.history: history of trained model
        '''

        try:

            model = self.build()
            no_errors = self.define_train_strategies() #THIS FUNCTION IS APPLIED ON INHERITED OBJECTS OF THIS CLASS (ALEX_NET OR VGG NET)
            if no_errors == False:
                raise
            history, model = self.train(model)
            predictions = self.predict(model)

            return model, predictions, history
        except:
            raise CustomError.ErrorCreationModel(config.ERROR_MODEL_EXECUTION)

    @abstractmethod
    def build(self, trainedModel=None) -> Sequential:
        pass

    @abstractmethod
    def train(self, model : Sequential) -> Tuple[History, Sequential]:
        pass

    def predict(self, model : Sequential):

        try:

            predictions = model.predict(
                x=self.X_test,
                use_multiprocessing=config.MULTIPROCESSING
            )

            #CHECK PREDICTIONS OUTPUT WITH REAL TARGETS
            argmax_preds = np.argmax(predictions, axis=1) #BY ROW, BY EACH SAMPLE

            return argmax_preds

        except:
            raise