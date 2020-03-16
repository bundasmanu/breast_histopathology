from . import Strategy
from imblearn.over_sampling import RandomOverSampler
from exceptions import CustomError
import config
import numpy as np
import keras
from sklearn.utils import shuffle
#REF: https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

class OverSampling(Strategy.Strategy):

    def __init__(self):
        pass

    def applyStrategy(self, X_train, y_train, **kwargs):

        '''
        THIS FUNCTION APPLIES UNDERSAMPLING TECHNIQUE ON TRAINING DATA
        :param X_train: numpy array --> training data
        :param y_train: numpy array --> training targets
        :return X_train: numpy array --> under sampler training data
        :return y_train: numpy array --> under sampler training targets
        '''

        try:

            if not bool(kwargs) == False:  # CHECK IF DICT IS EMPTY
                raise CustomError.ErrorCreationModel(config.ERROR_NO_ARGS_ACCEPTED)
                return

            numberValues = [np.argmax(y_train, axis=1)]
            numberValues = np.array(numberValues)
            numberValues = numberValues.reshape(numberValues.shape[0] * numberValues.shape[1])
            occorrences_counter = np.bincount(numberValues)
            print("\nNumber samples Class 0: ", occorrences_counter[0])
            print("\nNumber samples Class 1: ", occorrences_counter[1])

            overSampler = RandomOverSampler(random_state=0)  # ALLOWS REPRODUCIBILITY

            # I NEED TO RESHAPE TRAINING DATA TO 2D ARRAY (SAMPLES, FEATURES)
            X_train = self.reshape4D_to_2D(X_train)

            # APLY DECODE OF TARGET DATA NEEDED TO APPLY RESAMPLE
            decoded_ytrain = self.decodeYData(y_train)

            # APPLY RESAMPLE OF DATA
            X_train, decoded_ytrain = overSampler.fit_resample(X_train, decoded_ytrain)

            # I NEED TO RESHAPE DATA AGAIN FROM 2D TO 4D
            X_train = self.reshape2D_to_4D(X_train)

            occorrences_counter = np.bincount(decoded_ytrain)
            print("\nNumber samples Class 0: ", occorrences_counter[0])
            print("\nNumber samples Class 1: ", occorrences_counter[1])

            # TRANSFORM Y_DECODED TO CATEGORICAL AGAIN
            decoded_ytrain = keras.utils.to_categorical(decoded_ytrain, config.NUMBER_CLASSES)

            # SHUFFLE DATA
            X_train, decoded_ytrain = shuffle(X_train, decoded_ytrain)

            return X_train, decoded_ytrain

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_UNDERSAMPLING)