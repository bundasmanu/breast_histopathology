from . import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Dense, Flatten
import config
from exceptions import CustomError
from .Strategies_Train import UnderSampling, OverSampling
from keras.optimizers import Adam
from keras.callbacks.callbacks import History
from typing import Tuple

class AlexNet(Model.Model):

    def __init__(self, *args):
        super(AlexNet, self).__init__(*args)
    
    def define_train_strategies(self, undersampling=True, oversampling=False, data_augmentation=False) -> bool:
        return super(AlexNet, self).define_train_strategies(undersampling, oversampling, data_augmentation)
    
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
            model.summary()

            return model

        except:
            raise

    def train(self, model : Sequential) -> Tuple[History, Sequential]:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR MAKE THE TRAINING OF MODEL
        :param model: Sequential model builded before, or passed (already trained model)
        :return: Sequential model --> trained model
        :return: History.history --> train and validation loss and metrics variation along epochs
        '''

        try:

            if model is None:
                raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)
                return None

            # OPTIMIZER
            opt = Adam(learning_rate=config.LEARNING_RATE, decay=config.DECAY)

            # COMPILE
            model.compile(optimizer=opt, loss=config.LOSS_BINARY, metrics=[config.ACCURACY_METRIC])

            #GET STRATEGIES RETURN DATA, AND IF DATA_AUGMENTATION IS APPLIED TRAIN GENERATOR
            train_generator = None

            if len(self.StrategyList) == 0: #IF USER DOESN'T PRETEND EITHER UNDERSAMPLING AND OVERSAMPLING
                X_train = self.X_train
                y_train = self.y_train

            else: #USER WANTS AT LEAST UNDERSAMPLING OR OVERSAMPLING
                X_train, y_train = self.StrategyList[0].applyStrategy(self.X_train, self.y_train)
                if len(self.StrategyList) > 1: #USER CHOOSE DATA AUGMENTATION OPTION
                    train_generator = self.StrategyList[1].applyStrategy(self.X_train, self.y_train)

            if train_generator is None: #NO DATA AUGMENTATION

                history = model.fit(
                    x=X_train,
                    y=y_train,
                    batch_size=config.BATCH_SIZE_ALEX_NO_AUG,
                    epochs=config.EPOCHS,
                    validation_data=(self.X_val, self.y_val),
                    shuffle=True,
                    use_multiprocessing=config.MULTIPROCESSING
                )

                return history, model

            #ELSE APPLY DATA AUGMENTATION

            history = model.fit_generator(
                generator=train_generator,
                validation_data=(self.X_train, self.y_val),
                epochs=config.EPOCHS,
                steps_per_epoch=self.X_train.shape[0] / config.BATCH_SIZE_ALEX_AUG,
                shuffle=True,
                use_multiprocessing=config.MULTIPROCESSING
            )

            return history, model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_TRAINING)