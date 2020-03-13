import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import cv2
import os
import config_func
from sklearn.utils import shuffle

def main():

    #NUMBER PATIENTS
    numberPatients = os.listdir("../breast_histopathology/input/breast-histopathology-images/IDC_regular_ps50_idx5/")
    print(len(numberPatients))

    #GET ALL IMAGES
    images = glob(pathname='../breast_histopathology/input/breast-histopathology-images/IDC_regular_ps50_idx5/**/*.png',
                  recursive=True)  # RELATIVE PATHNAME
    print(len(images))

    # CREATION OF DATAFRAME WITH ALL IMAGES --> [ID_PATIENT, PATH_IMAGE, TARGET]
    data = pd.DataFrame(index=np.arange(0, len(images)), columns=["id", "image_path", "target"])

    patients_dirs = os.listdir("../breast_histopathology/input/breast-histopathology-images/IDC_regular_ps50_idx5")
    rootBase = "../breast_histopathology/input/breast-histopathology-images/IDC_regular_ps50_idx5"

    # POPULATE DATAFRAME
    add_row = 0
    for i in range(len(numberPatients)):
        patient_dir = patients_dirs[i]
        patient_link = os.path.join(rootBase, patient_dir)
        for path in os.listdir(os.path.join(patient_link)):
            files = os.path.join(patient_link, path)
            for file in os.listdir(files):
                data.iloc[add_row]["id"] = patient_dir
                data.iloc[add_row]["target"] = path
                data.iloc[add_row]["image_path"] = os.path.join(files, file)
                add_row = add_row + 1

    #INFO DATAFRAME
    data.head(5)
    data.shape
    data.info()

    #TRANSFORM DATA INTO NUMPY ARRAY'S
    X, Y = config_func.resize_images(50,50, data)

    #SHUFFLE DATA
    X, Y = shuffle(X, Y)

if __name__ == "__main__":
    main()