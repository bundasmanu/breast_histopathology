from . import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Dense, Flatten
import config
from exceptions import NoModel, IncoherentStrategy
from .Strategies_Train import UnderSampling, OverSampling

class AlexNet(Model):

    def __init__(self, underSampling=True, oversampling=False, dataAugmentation=False):
        if underSampling == True and oversampling == True:
            raise IncoherentStrategy
        if underSampling == True:
            self.under_sampling = UnderSampling()
        else:
            self.over_sampling = OverSampling()


    def build(self, trainedModel=None) -> Sequential:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR THE INITIALIZATION OF SEQUENTIAL ALEXNET MODEL
        :return: Sequential: AlexNet MODEL
        '''

        try:

            #IF USER ALREADY HAVE A TRAINED MODEL, AND NO WANTS TO BUILD AGAIN A NEW MODEL
            if trainedModel != None:
                return trainedModel

            model = Sequential()

            input_shape = (config.WIDTH, config.HEIGHT, config.CHANNELS)
            model.add(Conv2D(filters=16, input_shape=input_shape, kernel_size=(5,5), strides=1, padding=config.VALID_PADDING))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
            model.add(BatchNormalization())

            model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding='valid'))
            model.add(BatchNormalization())

            model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding=config.SAME_PADDING))
            model.add(BatchNormalization())

            model.add(Flatten())

            model.add(Dense(units=32))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(Dropout(0.5))

            model.add(Dense(units=16))
            model.add(Activation(config.RELU_FUNCTION))
            #DOESNT MAKE SENSE MAKE DROPOUT TO OPTPUT LAYER

            model.add(Dense(units=config.NUMBER_CLASSES))
            model.add(Activation(config.SOFTMAX_FUNCTION))

            return model

        except:
            raise

    def train(self, model : Sequential):

        '''
        THIS FUNCTION IS RESPONSIBLE FOR MAKE THE TRAINING OF MODEL
        :param model: Sequential model builded before, or passed (already trained model)
        :return: Sequential model --> trained model
        '''

        try:

            if model is None:
                raise NoModel



        except:
            raise