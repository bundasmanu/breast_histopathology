from . import Model
import Data
from exceptions import CustomError
import config
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Dense, Flatten, BatchNormalization, Input
from keras.callbacks.callbacks import History
from typing import Tuple
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
import config_func
import numpy
from .Strategies_Train import Strategy
from keras import regularizers
from keras.models import Model as mp
from models.Strategies_Train import DataAugmentation, UnderSampling, OverSampling

class VGGNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(VGGNet, self).__init__(data, *args)

    def addStrategy(self, strategy : Strategy.Strategy) -> bool:
        return self.StrategyList.append(strategy)

    def add_stack(self, input, numberFilters, dropoutRate, input_shape=None):

        '''
        This function represents the implementation of stack cnn layers (in this case using only 2 cnn layers compacted)
        Conv --> Activation --> Conv --> Activation --> MaxPooling --> BatchNormalization --> Dropout
        :param input: tensor with current model architecture
        :param numberFilters: integer: number of filters to put on Conv layer
        :param dropoutRate: float (between 0.0 and 1.0)
        :param input_shape: tuple (height, width, channels) with shape of first cnn layer --> default None (not initial layers)
        :return: tensor of updated model
        '''

        try:

            if input_shape!=None:
                input = Conv2D(filters=numberFilters, kernel_size=(3, 3), strides=1, input_shape=input_shape, kernel_initializer='he_uniform',
                       padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            else:
                input = Conv2D(filters=numberFilters, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING, kernel_initializer='he_uniform',
                           kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            input = Activation(config.RELU_FUNCTION) (input)
            input = Conv2D(filters=numberFilters, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING, kernel_initializer='he_uniform',
                           kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            input = Activation(config.RELU_FUNCTION) (input)
            input = BatchNormalization() (input)
            input = MaxPooling2D(pool_size=(2,2), strides=2, padding='same') (input)
            input = Dropout(dropoutRate) (input)

            return input

        except:
            raise

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

            # definition of input shape and Input Layer
            input_shape = (config.WIDTH, config.HEIGHT, config.CHANNELS)
            input = Input(input_shape)

            ## add stack conv layer to the model
            numberFilters = args[1]
            model = None
            for i in range(args[0]):
                if i == 0:
                    model = self.add_stack(input, numberFilters, 0.15, input_shape)
                else:
                    model = self.add_stack(model, numberFilters, 0.25)
                numberFilters += args[2]

            # flatten
            model = Flatten() (model)

            ## Full Connected Layers
            for i in range(args[3]):
                model = Dense(units=args[4], kernel_regularizer=regularizers.l2(config.DECAY), kernel_initializer='he_uniform') (model)
                model = Activation(config.RELU_FUNCTION) (model)
                model = BatchNormalization() (model)
                if i != (args[3] - 1):
                    model = Dropout(0.25) (model) ## applies Dropout on all FCL's except FCL preceding the ouput layer (sigmoid)

            # Output Layer (sigmoid)
            model = Dense(units=config.NUMBER_CLASSES) (model)
            model = Activation(config.SIGMOID_FUNCTION) (model)

            # build model
            model = mp(inputs=input, outputs=model)

            if config.BUILD_SUMMARY == 1:
                model.summary()

            return model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_BUILD)

    def train(self, model : Sequential, batch_size) -> Tuple[History, Sequential]:

        try:

            if model is None:
                raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)

            # OPTIMIZER
            opt = Adam(learning_rate=config.LEARNING_RATE, decay=config.DECAY)

            # COMPILE
            model.compile(optimizer=opt, loss=config.LOSS_BINARY, metrics=[config.ACCURACY_METRIC])

            #GET STRATEGIES RETURN DATA, AND IF DATA_AUGMENTATION IS APPLIED TRAIN GENERATOR
            train_generator = None

            # get data
            X_train = self.data.X_train
            y_train = self.data.y_train

            if self.StrategyList: # if strategylist is not empty
                for i, j in zip(self.StrategyList, range(len(self.StrategyList))):
                    if isinstance(i, DataAugmentation.DataAugmentation):
                        train_generator = self.StrategyList[j].applyStrategy(self.data)
                    if isinstance(i, OverSampling.OverSampling):
                        X_train, y_train = self.StrategyList[j].applyStrategy(self.data)
                    if isinstance(i, UnderSampling.UnderSampling):
                        X_train, y_train = self.StrategyList[j].applyStrategy(self.data)

            #reduce_lr = LearningRateScheduler(config_func.lr_scheduler)
            es_callback = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            decrease_callback = ReduceLROnPlateau(monitor='val_loss',
                                                        patience=1,
                                                        factor=0.7,
                                                        mode='min',
                                                        verbose=1,
                                                        min_lr=0.000001)
            decrease_callback2 = ReduceLROnPlateau(monitor='loss',
                                                        patience=1,
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
                    batch_size=batch_size,
                    epochs=config.EPOCHS,
                    validation_data=(self.data.X_val, self.data.y_val),
                    shuffle=True,
                    #use_multiprocessing=config.MULTIPROCESSING,
                    callbacks=[decrease_callback, decrease_callback2, es_callback],
                    #class_weight=class_weights,
                    verbose=config.TRAIN_VERBOSE
                )

                return history, model

            #ELSE APPLY DATA AUGMENTATION

            history = model.fit_generator(
                generator=train_generator,
                validation_data=(self.data.X_val, self.data.y_val),
                epochs=config.EPOCHS,
                steps_per_epoch=X_train.shape[0] / batch_size,
                shuffle=True,
                callbacks=[decrease_callback, decrease_callback2, es_callback],
                verbose=config.TRAIN_VERBOSE,
                class_weight=class_weights
            )

            return history, model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_TRAINING)

    def __str__(self):
        pass