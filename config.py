import numpy as np
import itertools

#global counter
counter_iterations = itertools.count(start=0, step=1)

# image dimensions
WIDTH = 50
HEIGHT = 50
CHANNELS = 3

ID = "id"
IMAGE_PATH = "image_path"
TARGET = "target"

PATIENTS_FOLDER_PATH = "../breast_histopathology/input/breast-histopathology-images/IDC_regular_ps50_idx5/"
CUSTOM_IMAGE_PATH = "../breast_histopathology/input/breast-histopathology-images/IDC_regular_ps50_idx5/**/*.png"

STANDARDIZE_AXIS_CHANNELS = (0,1,2,3)

NUMBER_CLASSES = 2

SIZE_DATAFRAME = 200000

VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2

ALEX_NET = "ALEXNET"
VGG_NET = "VGGNET"

PSO_OPTIMIZER = "PSO"
GA_OPTIMIZER = "GA"

RELU_FUNCTION = "relu"
SOFTMAX_FUNCTION = "softmax"

LEARNING_RATE = 0.01
DECAY = 1e-6

LOSS_BINARY = "binary_crossentropy"
LOSS_CATEGORICAL = "categorical_crossentropy"

VALID_PADDING = "valid"
SAME_PADDING = "same"

ACCURACY_METRIC = "accuracy"
VALIDATION_ACCURACY = "val_accuracy"

BATCH_SIZE_ALEX_NO_AUG = 128
BATCH_SIZE_ALEX_AUG = 128
EPOCHS = 15
MULTIPROCESSING = True
SHUFFLE = True

X_VAL_ARGS = "X_Val"
Y_VAL_ARGS = "y_val"

HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ROTATION_RANGE = 10

#FLAGS TRAIN STRATEGY
UNDERSAMPLING = True
OVERSAMPLING = False
DATA_AUGMENTATION = False

UNDERSAMPLING_KWARG_FLAG = "undersampling"
OVERSAMPLING_KWARG_FLAG = "oversampling"
DATA_AUGMENTATION_KWARG_FLAG = "data_augmentation"

#EXCEPTIONS MESSAGES
ERROR_MODEL_EXECUTION = "\nError on model execution"
ERROR_NO_ARGS = "\nPlease provide args: ",X_VAL_ARGS," and ", Y_VAL_ARGS
ERROR_NO_ARGS_ACCEPTED = "\nThis Strategy doesn't accept more arguments"
ERROR_NO_MODEL = "\nPlease pass a initialized model"
ERROR_INVALID_OPTIMIZER = "\nPlease define a valid optimizer: ", PSO_OPTIMIZER," or ", GA_OPTIMIZER
ERROR_INCOHERENT_STRATEGY = "\nYou cannot choose the oversampling and undersampling strategies at the same time"
ERROR_ON_UNDERSAMPLING = "\nError on undersampling definition"
ERROR_ON_OVERSAMPLING = "\nError on oversampling definition"
ERROR_ON_DATA_AUG = "\nError on data augmentation definition"
ERROR_ON_TRAINING = "\nError on training"
ERROR_ON_OPTIMIZATION = "\nError on optimization"
ERROR_INVALID_NUMBER_ARGS = "\nPlease provide correct number of args"
ERROR_ON_BUILD = "\nError on building model"
ERROR_APPEND_STRATEGY = "\nError on appending strategy"
ERROR_ON_PLOTTING = "\nError on plotting"

#PSO OPTIONS
PARTICLES = 20
ITERATIONS = 12
TOPOLOGY_FLAG = 1 # 0 MEANS GBEST, AND 1 MEANS LBEST
gbestOptions = {'w' : 0.9, 'c1' : 0.7, 'c2' : 0.7}
lbestOptions = {'w' : 0.9, 'c1' : 0.7, 'c2' : 0.7, 'k' : 4, 'p' : 2}

MAX_VALUES_LAYERS_ALEX_NET = [140, 140, 140, 256, 128, 256] # 4 conv's, 1 dense and batch size
MAX_VALUES_LAYERS_VGG_NET = [128, 128, 128, 196, 128, 256] # 4 conv's, 1 dense and batch size
MIN_BATCH_SIZE = 16

IDC_CLASS_NAME = "With IDC"
HEALTHY_CLASS_NAME = "Healthy"
LIST_CLASSES_NAME = [HEALTHY_CLASS_NAME, IDC_CLASS_NAME]

#GA OPTIONS
TOURNAMENT_SIZE = 100
INDPB = 0.6
CXPB = 0.4
MUTPB = 0.2

#FILENAMES SAVE MODELS
ALEX_NET_BEST_FILE = "alex_best.h5"
ALEX_NET_PSO_FILE = "alex_pso.h5"
ALEX_NET_GA_FILE = "alex_ga.h5"
VGG_NET_BEST_FILE = "vgg_best.h5"
VGG_NET_PSO_FILE = "vgg_pso.h5"
VGG_NET_GA_FILE = "vgg_ga.h5"

#FILENAME POSITION PSO VARIATION
POS_VAR_LOWER = 'particlesPso.mp4'
POS_VAR_INTER = 'particlesPso_intermedia.mp4'
POS_VAR_HIGHTER = 'particlesPso_elevada.mp4'
POS_VAR_EXP = 'pos_var_exp.html'

#NAMES DIMENSIONS PSO --> array
DIMENSIONS_NAMES = ['1 Conv', '2 Conv', '3 Conv', '4 Conv', 'Dense', 'Batch']

# VARIABLES MAKE .mp4 VIDEO with particles movement position
X_LIMITS = [1, 128]
Y_LIMITS = [1, 140]
LABEL_X_AXIS = 'Nºfiltros 1ªcamada'
LABEL_Y_AXIS = 'Nºfiltros 2ªcamada'

# PSO INIT DEFINITIONS --> IN ARGS FORM
pso_init_args_alex = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    6,  # dimensions (6 conv filters, 1 dense neurons and batch size)
    np.array(MAX_VALUES_LAYERS_ALEX_NET)  # superior bound limits for dimensions
)

pso_init_args_vgg = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    6,  # dimensions (6 conv filters, 1 dense neurons and batch size)
    np.array(MAX_VALUES_LAYERS_VGG_NET)  # superior bound limits for dimensions
)