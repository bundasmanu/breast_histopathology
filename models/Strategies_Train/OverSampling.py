from . import Strategy
from imblearn.over_sampling import RandomOverSampler
from exceptions import CustomError
import config
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

            if not bool(kwargs) == False: #CHECK IF DICT IS EMPTY
                raise CustomError.ErrorCreationModel(config.ERROR_NO_ARGS_ACCEPTED)
                return

            overSampler = RandomOverSampler(random_state=0) #ALLOWS REPRODUCIBILITY

            X_train, y_train = overSampler.fit_sample(X_train, y_train)

            return X_train, y_train

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OVERSAMPLING)