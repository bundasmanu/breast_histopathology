import numpy as np
import pandas as pd
import config_func
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import config
from models import ModelFactory
import Data
from optimizers import OptimizerFactory, Optimizer, PSO
from models.Strategies_Train import UnderSampling, Strategy, DataAugmentation
import matplotlib.pyplot as plt
from keras.models import load_model
import keras
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #MAKES MORE FASTER THE INITIAL SETUP OF GPU --> WARNINGS INITIAL STEPS IS MORE QUICKLY
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #THIS LINE DISABLES GPU OPTIMIZATION

def main():

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '    DATA PREPARATION (PRE-PROCESSING, CLEAN, TRANSFORM)  '
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    print("#################","DATA PREPARATION","####################\n")
    # CREATION OF DATAFRAME WITH ALL IMAGES --> [ID_PATIENT, PATH_IMAGE, TARGET]
    data = pd.DataFrame(index=np.arange(0, config.SIZE_DATAFRAME), columns=[config.ID, config.IMAGE_PATH, config.TARGET])

    # POPULATE DATAFRAME
    data = config_func.populate_DataFrame(data)

    #TRANSFORM DATA INTO NUMPY ARRAY'S
    X, Y = config_func.resize_images(config.WIDTH,config.HEIGHT, data)

    #DIVISION OF DATASET'S BETWEEN TRAIN, VALIDATION AND TEST --> I NEED ATTENTION, BECAUSE CLASSES ARE UNBALANCED
    indexes = np.arange(X.shape[0])
    X_train, X_val, y_train, y_val, indeces_train, indices_val = train_test_split(X, Y, indexes, test_size=config.VALIDATION_SIZE,
                                                shuffle=True, random_state=config.RANDOM_STATE) #RANDOM STATE IS NEEDED TO GUARANTEES REPRODUCIBILITY
    indexes = indeces_train
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_train, y_train, indexes, test_size=config.TEST_SIZE,
                                                                    shuffle=True, random_state=config.RANDOM_STATE)
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    #NORMALIZE DATA
    X_train, X_val, X_test = config_func.normalize(X_train, X_val, X_test)

    #ONE HOT ENCODING TARGETS
    y_train, y_val, y_test = config_func.one_hot_encoding(y_train, y_val, y_test)
    print("#################", "DATA PREPARATION CONCLUDED", "####################\n")

    #CREATE OBJECT DATA
    d = Data.Data(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test
    )

    factoryModel = ModelFactory.ModelFactory()
    numberLayers = (
        4, #CNN LAYERS
        1 #DENSE LAYERS
    )

    ## STRATEGIES OF TRAIN INSTANCES

    underSampling = UnderSampling.UnderSampling()
    data_aug = DataAugmentation.DataAugmentation()

    ## ---------------------------ALEXNET APPLICATION ------------------------------------

    ## DICTIONARIES DEFINITION
    numberLayers = (
        4, #CNN LAYERS
        1 #DENSE LAYERS
    )

    valuesLayers = (
        1, ## number of normal convolutional layers (init conv doen't count here, because always exist)
        2, ## number of stacked cnn layers
        16, ## number of feature maps of first conv layer
        16, ## growth rate
        2, ## number of FCL's preceding output layer (sigmoid layer)
        16, ## number of neurons of Full Connected Layer
        config.BATCH_SIZE_ALEX_AUG #batch size
    )

    # CREATION OF MODEL
    alexNetModel = factoryModel.getModel(config.ALEX_NET, d, *numberLayers)

    ## APPLY STRATEGIES OF TRAIN
    alexNetModel.addStrategy(underSampling)
    alexNetModel.addStrategy(data_aug)

    #model, predictions, history = alexNetModel.template_method(*valuesLayers)

    #config_func.print_final_results(d.y_test, predictions, history)

    ## ---------------------------VGGNET APPLICATION ------------------------------------

    ## DICTIONARIES DEFINITION
    numberLayers = (
        4, #CNN LAYERS
        1 #DENSE LAYERS
    )

    valuesLayers = (
        4, # 4 stacks (5 in total, because init stack is only needed)
        16, # number of feature maps of initial convolution layer
        16, # growth rate
        2, ## number of FCL's preceding output layer (sigmoid layer)
        16, # number neurons of Full Connected Layer
        config.BATCH_SIZE_ALEX_AUG # batch size
    )

    vggNetModel = factoryModel.getModel(config.VGG_NET, d, *numberLayers)

    vggNetModel.addStrategy(underSampling)
    vggNetModel.addStrategy(data_aug)

    #model, predictions, history = vggNetModel.template_method(*valuesLayers)

    #config_func.print_final_results(d.y_test, predictions, history)

    ## ---------------------------DENSENET APPLICATION ------------------------------------

    # # DICTIONARIES DEFINITION
    numberLayers = (
        4, #BLOCKS
        1 #DENSE LAYERS
    )

    valuesLayers = (
        16, # initial number of Feature Maps
        4, # number of dense blocks
        2, # number of layers in each block
        16, # growth rate
        config.BATCH_SIZE_ALEX_AUG # batch size
    )

    densenet = factoryModel.getModel(config.DENSE_NET, d, *numberLayers)

    densenet.addStrategy(underSampling)
    densenet.addStrategy(data_aug)

    #model, predictions, history = densenet.template_method(*valuesLayers)

    #config_func.print_final_results(d.y_test, predictions, history)

    ## ------------------------PSO OPTIMIZATION ------------------------------------------

    #PSO OPTIMIZATION
    optFact = OptimizerFactory.OptimizerFactory()

    # definition optimizers for models
    pso_alex = optFact.createOptimizer(config.PSO_OPTIMIZER, alexNetModel, *config.pso_init_args_alex)
    pso_vgg = optFact.createOptimizer(config.PSO_OPTIMIZER, vggNetModel, *config.pso_init_args_vgg)
    pso_dense = optFact.createOptimizer(config.PSO_OPTIMIZER, densenet, *config.pso_init_args_densenet)

    # call optimize function
    cost, pos, optimizer = pso_dense.optimize()

    #plot cost history and plot position history
    print("Custo: {}".format(cost))
    config_func.print_Best_Position_PSO(pos, config.DENSE_NET) # print position
    pso_dense.plotCostHistory(optimizer=optimizer)
    pso_dense.plotPositionHistory(optimizer, np.array(config.X_LIMITS), np.array(config.Y_LIMITS),
                                 config.POS_VAR_EXP, config.LABEL_X_AXIS, config.LABEL_Y_AXIS)

    ## --------------------------ENSEMBLE ---------------------------------------------------

    # # load models, that are saved in files
    # alexNetModel = load_model(config.ALEX_NET_BEST_FILE)
    # vggNetModel = load_model(config.VGG_NET_BEST_FILE)
    #
    # # list of models to ensemble
    # ensemble_models = [alexNetModel, vggNetModel]
    #
    # # get ensemble model
    # ensemble_model = config_func.ensemble(ensemble_models)
    #
    # # predict using ensemble model
    # predictions = ensemble_model.predict(d.X_test)
    # argmax_preds = np.argmax(predictions, axis=1)  # BY ROW, BY EACH SAMPLE
    # predictions = keras.utils.to_categorical(argmax_preds)
    #
    # # print final results of predict using ensemble model (report and confusion matrix)
    # config_func.print_final_results(y_test=d.y_test, predictions=predictions, history=None)
    #
    # # save ensemble model
    # ensemble_model.save(config.ENSEMBLE_NORMAL_MODEL)
    # del ensemble_model

if __name__ == "__main__":
    main()