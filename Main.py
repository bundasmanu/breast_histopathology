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
from models.Strategies_Train import UnderSampling, Strategy
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #MAKES MORE FASTER THE INITIAL SETUP OF GPU --> WARNINGS INITIAL STEPS IS MORE QUICKLY
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
        2 #DENSE LAYERS
    )

    valuesLayers = (
        32,
        48,
        64,
        64,
        16,
        8
    )

    #CREATE MODEL
    alexNetModel = factoryModel.getModel(config.ALEX_NET, d, *numberLayers)

    #CREATE STRATEGIES AND PASS TO MODEL --> if i don't want to train using strategies, i didn't add strategies
    #underSampling = UnderSampling.UnderSampling() #or d variable
    #alexNetModel.addStrategy(underSampling)
    model, predictions, history = alexNetModel.template_method(*valuesLayers)
    print(config_func.plot_cost_history(history))
    print(config_func.plot_accuracy_plot(history))
    predictions = config_func.decode_array(predictions) #DECODE ONE-HOT ENCODING PREDICTIONS ARRAY
    y_test_decoded = config_func.decode_array(alexNetModel.data.y_test)  # DECODE ONE-HOT ENCODING y_test ARRAY
    report, confusion_mat = config_func.getConfusionMatrix(y_test_decoded, predictions)
    print(report)
    plt.figure()
    config_func.plot_confusion_matrix(confusion_mat, config.LIST_CLASSES_NAME)

    #PSO OPTIMIZATION
    # optFact = OptimizerFactory.OptimizerFactory()
    # opt_options = (
    #     config.PARTICLES,
    #     config.ITERATIONS,
    #     config.PSO_DIMENSIONS
    # )
    # optPSO = optFact.createOptimizer(config.PSO_OPTIMIZER, alexNetModel, *opt_options)
    # cost, pos = optPSO.optimize()
    # print(cost)
    # print(pos)

if __name__ == "__main__":
    main()