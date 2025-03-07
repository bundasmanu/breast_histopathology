from . import Strategy
from keras.preprocessing.image import ImageDataGenerator
import config
from exceptions import CustomError
import Data

class DataAugmentation(Strategy.Strategy):

    def __init__(self):
        super(DataAugmentation, self).__init__()

    def applyStrategy(self, data : Data.Data, **kwargs):

        '''
        THIS FUNCTION IS RESPONSIBLE TO FIT MODEL USING DATA AUGMENTATION
        :param X_train: training data
        :param y_train: training targets
        :param kwargs:
        :return: train_generator: tuple (augmented X, augmented Y)
        '''

        try:

            image_gen = ImageDataGenerator(
                horizontal_flip=config.HORIZONTAL_FLIP,
                vertical_flip=config.VERTICAL_FLIP,
                width_shift_range=config.WIDTH_SHIFT_RANGE,
                height_shift_range=config.HEIGHT_SHIFT_RANGE,
            )

            image_gen.fit(data.X_train, augment=True) #DATA AUGMENTATION

            train_generator = image_gen.flow(
                data.X_train,
                data.y_train,
                batch_size=config.BATCH_SIZE_ALEX_AUG,
            )

            return train_generator

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_DATA_AUG)