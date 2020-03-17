import numpy as np
import pandas as pd
import config_func
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import config
from models import Model, ModelFactory, AlexNet
import Data
from optimizers import OptimizerFactory, Optimizer, PSO

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
        16,
        32,
        64,
        64,
        32,
        16
    )
    alexNetModel = factoryModel.getModel(config.ALEX_NET, *numberLayers)
    #model, predictions, history = alexNetModel.template_method(d, *valuesLayers)
    #print(predictions.shape)
    #print(y_test[0])
    #print(predictions[0])

    #PSO OPTIMIZATION
    optFact = OptimizerFactory.OptimizerFactory()
    opt_options = (
        config.PARTICLES,
        config.ITERATIONS,
        config.PSO_DIMENSIONS
    )
    optPSO = optFact.createOptimizer(config.PSO_OPTIMIZER, alexNetModel, *opt_options)
    cost, pos = optPSO.optimize(d)

if __name__ == "__main__":
    main()