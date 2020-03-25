from . import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Dense, Flatten
import config
from exceptions import CustomError
from .Strategies_Train import Strategy
from keras.optimizers import Adam
from keras.callbacks.callbacks import History
from typing import Tuple
import Data
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import config_func
from sklearn.utils import class_weight
import numpy

class AlexNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(AlexNet, self).__init__(data, *args)

    def addStrategy(self, strategy : Strategy.Strategy) -> bool:
        return super(AlexNet, self).addStrategy(strategy)
    
    def build(self, *args, trainedModel=None) -> Sequential:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR THE INITIALIZATION OF SEQUENTIAL ALEXNET MODEL
        :param args: list integers, in logical order --> to populate cnn (filters) and dense (neurons)
        :return: Sequential: AlexNet MODEL
        '''

        try:

            #IF USER ALREADY HAVE A TRAINED MODEL, AND NO WANTS TO BUILD AGAIN A NEW MODEL
            if trainedModel != None:
                return trainedModel

            if len(args) < (self.nDenseLayers+self.nCNNLayers):
                raise CustomError.ErrorCreationModel(config.ERROR_INVALID_NUMBER_ARGS)

            model = Sequential()

            input_shape = (config.WIDTH, config.HEIGHT, config.CHANNELS)
            model.add(Conv2D(filters=args[0], input_shape=input_shape, kernel_size=(5,5), strides=1, padding=config.VALID_PADDING, kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
            model.add(BatchNormalization())
            model.add(Dropout(0.15))

            model.add(Conv2D(filters=args[1], kernel_size=(3,3), strides=1, padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(BatchNormalization())
            model.add(Dropout(0.15))

            model.add(Conv2D(filters=args[2], kernel_size=(3,3), strides=1, padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(Conv2D(filters=args[2], kernel_size=(3,3), strides=1, padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2,2), strides=2))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            model.add(Conv2D(filters=args[3], kernel_size=(3,3), strides=1, padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(Conv2D(filters=args[3], kernel_size=(3,3), strides=1, padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding=config.SAME_PADDING))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            model.add(Flatten())

            model.add(Dense(units=args[4], kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(Dropout(0.5))

            # model.add(Dense(units=args[5], kernel_regularizer=regularizers.l2(config.DECAY)))
            # model.add(Activation(config.RELU_FUNCTION))
            #DOESNT MAKE SENSE MAKE DROPOUT TO OPTPUT LAYER

            model.add(Dense(units=config.NUMBER_CLASSES))
            model.add(Activation(config.SOFTMAX_FUNCTION))
            model.summary()

            return model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_BUILD)

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
                X_train = self.data.X_train
                y_train = self.data.y_train

            else: #USER WANTS AT LEAST UNDERSAMPLING OR OVERSAMPLING
                X_train, y_train = self.StrategyList[0].applyStrategy(self.data)
                if len(self.StrategyList) > 1: #USER CHOOSE DATA AUGMENTATION OPTION
                    train_generator = self.StrategyList[1].applyStrategy(self.data)

            #reduce_lr = LearningRateScheduler(config_func.lr_scheduler)
            es_callback = EarlyStopping(monitor='loss', patience=6)
            # decrease_callback = ReduceLROnPlateau(monitor='val_loss',
            #                                             patience=3,
            #                                             factor=0.7,
            #                                             mode='min',
            #                                             verbose=1,
            #                                             min_lr=0.000001)

            decrease_callback2 = ReduceLROnPlateau(monitor='loss',
                                                        patience=2,
                                                        factor=0.7,
                                                        mode='min',
                                                        verbose=1,
                                                        min_lr=0.000001)

            weights_y_train = config_func.decode_array(y_train)
            class_weights = class_weight.compute_class_weight('balanced',
                                                              numpy.unique(weights_y_train),
                                                              weights_y_train)

            if train_generator is None: #NO DATA AUGMENTATION

                history = model.fit(
                    x=X_train,
                    y=y_train,
                    batch_size=config.BATCH_SIZE_ALEX_NO_AUG,
                    epochs=config.EPOCHS,
                    validation_data=(self.data.X_val, self.data.y_val),
                    shuffle=True,
                    use_multiprocessing=config.MULTIPROCESSING,
                    callbacks=[decrease_callback2, es_callback],
                    class_weight=class_weights
                )

                return history, model

            #ELSE APPLY DATA AUGMENTATION

            history = model.fit_generator(
                generator=train_generator,
                validation_data=(self.data.X_val, self.data.y_val),
                epochs=config.EPOCHS,
                steps_per_epoch=X_train.shape[0] / config.BATCH_SIZE_ALEX_AUG,
                shuffle=True,
                use_multiprocessing=config.MULTIPROCESSING,

            )

            return history, model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_TRAINING)

    def __str__(self):
        pass