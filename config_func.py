import cv2
import numpy
import config
import os
from glob import glob
import numpy as np
import keras
import random

def getNumberPatients():

    '''
    THIS FUNCTION CHECKS HOW MANY FOLDERS ARE INSIDE PATIENTS_FOLDER_PATH --> TOTAL OF PATIENTS
    :return: integer: number of patients
    '''

    try:

        numberPatients = os.listdir(config.PATIENTS_FOLDER_PATH) #LIST OF ALL FOLDERS ON PATIENTS FOLDER PATH
        return len(numberPatients)

    except:
        raise

def getNumberOfImages():

    '''
    THIS FUNCTION CHECKS HOW MANY FILES ARE PNG FILES, CONSIDERING CUSTOM IMAGES PATH
    :return: integer: total of images, that corresponds to custom path
    '''

    try:

        images = glob(pathname= config.CUSTOM_IMAGE_PATH, recursive=True)  # RELATIVE PATHNAME --> RETURN LIST OF FILES
        return len(images)

    except:
        raise

def populate_DataFrame(data):

    '''

    :param data: DataFrame to populate
    :return: dataframe: Populated DataFrame
    '''

    try:

        all_patients_index = [i for i in range(getNumberPatients())]
        sorted_patients_index = random.sample(all_patients_index, len(all_patients_index))

        add_row = 0
        patients_dirs = os.listdir(config.PATIENTS_FOLDER_PATH)
        for i in sorted_patients_index:
            patient_dir = patients_dirs[i]
            patient_link = os.path.join(config.PATIENTS_FOLDER_PATH, patient_dir)
            for path in os.listdir(os.path.join(patient_link)):
                files = os.path.join(patient_link, path)
                for file in os.listdir(files):
                    data.iloc[add_row][config.ID] = patient_dir
                    data.iloc[add_row][config.TARGET] = path
                    data.iloc[add_row][config.IMAGE_PATH] = os.path.join(files, file)
                    add_row = add_row + 1
                    if add_row == config.SIZE_DATAFRAME:
                        return data

        return data

    except:
        raise

def resize_images(width, height, data):

    '''

    :param width: int --> pixel width to resize image
    :param height: int --> pixel height to resize image
    :param data: dataframe --> shape ["id", "image_path", "target"]
    :return x: numpy array --> shape (number images, width, height)
    :return y: numpy array --> shape (number images, target)
    '''

    try:

        x = []
        y = []

        for i in range(len(data[config.ID])):
            image = cv2.imread(data.at[i, config.IMAGE_PATH])
            x.append(cv2.resize(image, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_CUBIC))
            y.append(data.at[i, config.TARGET])

        return numpy.array(x), numpy.array(y)

    except:
        raise

def normalize(X_train, X_val, X_test):

    '''
    #REF https://forums.fast.ai/t/images-normalization/4058/8
    :param X_train: numpy array representing training data
    :param X_val: numpy array representing validation data
    :param X_test: numpy array representing test data
    :return X_train: numpy array normalized
    :return X_val: numpy array normalized
    :return X_X_test: numpy array normalized
    '''

    try:

        mean = np.mean(X_train,axis=config.STANDARDIZE_AXIS_CHANNELS) #STANDARDIZE BY CHANNELS
        std = np.std(X_train, axis=config.STANDARDIZE_AXIS_CHANNELS) #STANDARDIZE BY CHANNELS
        X_train = (X_train-mean)/(std+1e-7)
        X_val = (X_val-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)

        return X_train, X_val, X_test
    except:
        raise

def one_hot_encoding(y_train, y_val, y_test):

    '''

    :param y_train: numpy array with training targets
    :param y_val: numpy array with validation targets
    :param y_test: numpy array with test targets
    :return y_train: numpy array categorized [1 0] --> class 0 or [0 1] --> class 1
    :return y_val: numpy array categorized
    :return y_test: numpy array categorized
    '''

    try:

        y_train = keras.utils.to_categorical(y_train, config.NUMBER_CLASSES)
        y_val =  keras.utils.to_categorical(y_val, config.NUMBER_CLASSES)
        y_test =  keras.utils.to_categorical(y_test, config.NUMBER_CLASSES)

        return y_train, y_val, y_test

    except:
        raise