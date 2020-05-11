from . import Model
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Activation, Dropout, BatchNormalization, Dense, concatenate, Input, ZeroPadding2D, AveragePooling2D, MaxPooling2D
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
import numpy as np
from keras.models import Model as mp
from keras_applications.densenet import DenseNet as dnet

class DenseNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(DenseNet, self).__init__(data, *args)

    def addStrategy(self, strategy : Strategy.Strategy) -> bool:
        return super(DenseNet, self).addStrategy(strategy)

    def H(self, inputs, num_filters):

        '''
        THIS FUNCTION SIMULES THE BEHAVIOR OF EXPRESSION PRESENT IN PAPER:
            - xl=H(xl−1)+xl−1
                * H:  represents a composite function which takes in an image/feature map ( x ) and performs some operations on it.
                * x → Batch Normalization → ReLU → Zero Padding → 3×3 Convolution → Dropout
        :param inputs: previous layer
        :param num_filters: integer: number of filters of CNN layer
        :return: Convolution Layer output
        '''

        conv_out = BatchNormalization(epsilon=1.1e-5)(inputs)
        conv_out = Activation(config.RELU_FUNCTION)(conv_out)
        bootleneck_filters = num_filters * 2 ## paper
        conv_out = Conv2D(bootleneck_filters, kernel_size=(1, 1), use_bias=False, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv_out)

        conv_out = BatchNormalization(epsilon=1.1e-5)(conv_out)
        conv_out = Activation(config.RELU_FUNCTION)(conv_out)
        #conv_out = ZeroPadding2D((1, 1))(conv_out)
        conv_out = Conv2D(num_filters, kernel_size=(3, 3), use_bias=False, padding=config.SAME_PADDING, kernel_initializer='he_normal')(conv_out)
        #conv_out = Dropout(0.15)(conv_out)
        return conv_out

    def transition(self, inputs):

        '''
        The Transition layers perform the downsampling of the feature maps. The feature maps come from the previous block.
        :param inputs: previous layer
        :return: Convolution Layer output
        '''

        x = BatchNormalization(epsilon=1.1e-5)(inputs)
        x = Activation(config.RELU_FUNCTION)(x)
        num_feature_maps = inputs.shape[1].value

        x = Conv2D(filters=np.floor( 0.5 * num_feature_maps ).astype( np.int ),
                                   kernel_size=(1, 1), use_bias=False, padding=config.SAME_PADDING, kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(1e-4))(x)
        #x = Dropout(rate=0.15)(x)

        x = AveragePooling2D(pool_size=(2, 2))(x)
        return x

    def dense_block(self, inputs, num_layers, initFilter, growth_rate):

        '''
        This function represents the logic of Dense block
        :param inputs: input from previous Transition layer
        :param num_layers: integer : number of Conv layer
        :param initFilter: integer: initial number of filters of Bootleneck layer (H function)
        :param growth_rate: integer: increase of filters on bootleneck layers
        :return: Concatenation of Bootleneck layers
        '''

        try:

            for i in range(num_layers):
                conv_outputs = self.H(inputs, initFilter)
                inputs = concatenate([conv_outputs, inputs])
                initFilter += growth_rate
            return inputs, initFilter

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

            input_shape = (config.WIDTH, config.HEIGHT, config.CHANNELS)
            input = Input(shape=(input_shape))

            x = Conv2D(args[0], kernel_size=(3, 3), use_bias=False, kernel_initializer='he_normal',
                        strides=2, padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(1e-4))(input)
            x = MaxPooling2D(pool_size=(2,2), strides=2)(x)

            nFilters = 16
            for i in range(args[1]):
                x, nFilters = self.dense_block(x, args[2], nFilters, args[3])
                if i < (args[1] - 1):
                    x = self.transition(x) ## in last block (final step doesn't apply transition logic, global average pooling, made thiis

            x = GlobalAveragePooling2D()(x)

            x = Dense(config.NUMBER_CLASSES)(x)  # Num Classes for CIFAR-10
            outputs = Activation(config.SIGMOID_FUNCTION)(x)

            model = mp(input, outputs)

            if config.BUILD_SUMMARY == 1:
                model.summary()

            return model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_BUILD)

    def train(self, model : Sequential, batch_size) -> Tuple[History, Sequential]:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR MAKE THE TRAINING OF MODEL
        :param model: Sequential model builded before, or passed (already trained model)
        :return: Sequential model --> trained model
        :return: History.history --> train and validation loss and metrics variation along epochs
        '''

        try:

            if model is None:
                raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)

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
            es_callback = EarlyStopping(monitor='val_loss', patience=3)
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
                                                              np.unique(weights_y_train),
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
                    callbacks=[decrease_callback2, es_callback, decrease_callback],
                    class_weight=class_weights,
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
                callbacks=[decrease_callback2, es_callback, decrease_callback],
                verbose=config.TRAIN_VERBOSE
            )

            return history, model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_TRAINING)

    def __str__(self):
        pass