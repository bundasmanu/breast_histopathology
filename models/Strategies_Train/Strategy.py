from abc import ABC, abstractmethod
import numpy as np
import config

class Strategy(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def applyStrategy(self, X_train, y_train, **kwargs):
        pass

    def decodeYData(self, y_train):

        '''
        THIS FUNCTION IS USED TO DISABLE ONE-HOT-ENCODING
        e.g --> y_train = [ [0 1] [1 0] ]
                return [ [1] [0] ]
        :param y_train: numpy array: training targets
        :return: numpy array: decoded training targets
        '''

        try:

            decoded_ytrain = [np.argmax(y_train[i], axis=0) for i in range(y_train.shape[0])]
            decoded_ytrain = np.array(decoded_ytrain)

            return decoded_ytrain

        except:
            raise

    def reshape4D_to_2D(self, X_train):

        '''
        THIS FUNCTION IS USED TO RESHAPE TRAINING DATA FROM 4D TO 2D --> IS NEED TO APPLY STRATEGIES
        :param X_train: numpy array --> training data 4D (SAMPLES, WIDTH, HEIGHT, CHANNELS)
        :return: numpy array --> training data 2D (SAMPLES, FEATURES) --> FEATURES = (WIDTH * HEIGHT * CHANNELS)
        '''

        try:

            feature_reshape = (X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
            X_train = X_train.reshape(X_train.shape[0], feature_reshape)

            return X_train

        except:
            raise

    def reshape2D_to_4D(self, X_train):

        '''
        THIS FUNCTION IS USED TO RESHAPE TRAINING DATA FROM 2D TO 4D --> IS NEED TO APPLY STRATEGIES
        :param X_train: numpy array --> training data 2D (SAMPLES, FEATURES) --> FEATURES = (WIDTH * HEIGHT * CHANNELS)
        :return: numpy array --> training data 4D  (SAMPLES, WIDTH, HEIGHT, CHANNELS)
        '''

        try:

            shape_data = (X_train.shape[0], config.WIDTH, config.HEIGHT, config.CHANNELS)
            X_train = X_train.reshape(shape_data)

            return X_train

        except:
            raise