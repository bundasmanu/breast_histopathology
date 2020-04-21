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
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #THIS LINE DISABLES GPU OPTIMIZATION

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

    #SHUFFLE DATA
    X, Y = shuffle(X, Y)

    #DIVISION OF DATASET'S BETWEEN TRAIN, VALIDATION AND TEST --> I NEED ATTENTION, BECAUSE CLASSES ARE UNBALANCED
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=config.VALIDATION_SIZE, random_state=0, stratify=Y) #RANDOM STATE IS NEEDED TO GUARANTEES REPRODUCIBILITY
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=config.TEST_SIZE, random_state=0, stratify=y_train)
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

    valuesLayers = (
        16,
        24,
        48,
        72,
        12,
        config.BATCH_SIZE_ALEX_NO_AUG
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
        16,
        24,
        48,
        72,
        12,
        config.BATCH_SIZE_ALEX_NO_AUG
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
        16,
        24,
        48,
        72,
        12,
        config.BATCH_SIZE_ALEX_AUG
    )

    vggNetModel = factoryModel.getModel(config.VGG_NET, d, *numberLayers)

    vggNetModel.addStrategy(underSampling)
    vggNetModel.addStrategy(data_aug)

    #model, predictions, history = vggNetModel.template_method(*valuesLayers)

    #config_func.print_final_results(d.y_test, predictions, history)

    ## ------------------------PSO OPTIMIZATION ------------------------------------------

    # #PSO OPTIMIZATION
    optFact = OptimizerFactory.OptimizerFactory()

    # definition optimizers for models
    pso_alex = optFact.createOptimizer(config.PSO_OPTIMIZER, alexNetModel, *config.pso_init_args_alex)
    pso_vgg = optFact.createOptimizer(config.PSO_OPTIMIZER, vggNetModel, *config.pso_init_args_vgg)

    # call optimize function
    cost, pos, optimizer = pso_alex.optimize()

    #plot cost history and plot position history
    print(cost)
    print(pos)
    pso_alex.plotCostHistory(optimizer=optimizer)
    pso_alex.plotPositionHistory(optimizer, np.array(config.X_LIMITS), np.array(config.Y_LIMITS),
                                 config.POS_VAR_EXP, config.LABEL_X_AXIS, config.LABEL_Y_AXIS)

    ## --------------------------ENSEMBLE ---------------------------------------------------

    # # load models, that are saved in files
    # alexNetModel = load_model(config.ALEX_NET_BEST_FILE)
    # vggNetModel = load_model(config.VGG_NET_BEST_FILE)
    #
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

if __name__ == "__main__":
    main()