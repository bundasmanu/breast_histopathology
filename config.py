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

STANDARDIZE_AXIS_CHANNELS = (0,1,2)

NUMBER_CLASSES = 2

SIZE_DATAFRAME = 277524

VALIDATION_SIZE = 0.2
TEST_SIZE = 0.25

ALEX_NET = "ALEXNET"
VGG_NET = "VGGNET"
RES_NET = "RESNET"
DENSE_NET = "DENSENET"

PSO_OPTIMIZER = "PSO"
GA_OPTIMIZER = "GA"

RELU_FUNCTION = "relu"
SOFTMAX_FUNCTION = "softmax"
SIGMOID_FUNCTION = "sigmoid"

LEARNING_RATE = 0.001
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

RANDOM_STATE = 0
HE_SEED = 0

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
ERROR_ON_IDENTITY_BLOCK ="\nError on modelling identity block, please check the problem"
ERROR_ON_CONV_BLOCK ="\nError on modelling convolutional block, please check the problem"
ERROR_APPEND_STRATEGY = "\nError on appending strategy"
ERROR_ON_PLOTTING = "\nError on plotting"

#PSO OPTIONS
PARTICLES = 20
ITERATIONS = 12
TOPOLOGY_FLAG = 0 # 0 MEANS GBEST, AND 1 MEANS LBEST
gbestOptions = {'w' : 0.7, 'c1' : 1.4, 'c2' : 1.4}
lbestOptions = {'w' : 0.7, 'c1' : 1.4, 'c2' : 1.4, 'k' : 2, 'p' : 2}

MAX_VALUES_LAYERS_ALEX_NET = [3.99, 3.99, 128, 64, 2.99, 64, 128] # nº of normal conv's, nº of stack cnn layers, nº of feature maps of initial conv, growth rate, nº neurons of FCL layer and batch size
MIN_VALUES_LAYERS_ALEX_NET = [1, 1, 4, 0, 1, 8, 16]
MAX_VALUES_LAYERS_VGG_NET = [6.99, 128, 64, 2.99, 64, 128] # nº of stack cnn layers, nº of feature maps of initial conv, growth rate, nº neurons of FCL layer and batch size
MIN_VALUES_LAYERS_VGG_NET = [2, 4, 0, 1, 8, 16]
MAX_VALUES_LAYERS_RES_NET = [128, 2.99, 4.99, 64, 128] # number of filters of first conv layer, number of conv+identity blocks, growth rate and batch size
MIN_VALUES_LAYERS_RES_NET = [4, 0, 1, 0, 16]
MAX_VALUES_LAYERS_DENSE_NET = [128, 4.99, 6.99, 24, 1.0, 128] # nº of initial filters, nº of dense blocks, nº of composite blocks, growth rate, compression rate and batch size
MIN_VALUES_LAYERS_DENSE_NET = [4, 1, 2, 0, 0.1, 16]

IDC_CLASS_NAME = "With IDC"
HEALTHY_CLASS_NAME = "Healthy"
LIST_CLASSES_NAME = [HEALTHY_CLASS_NAME, IDC_CLASS_NAME]

#GA OPTIONS
TOURNAMENT_SIZE = 100
INDPB = 0.6
CXPB = 0.4
MUTPB = 0.2

# class weights
class_weights={
    0: 1.0, # healthy
    1: 2.5, # with idc
}

#FILENAMES SAVE MODELS
ALEX_NET_BEST_FILE = "alex_best.h5"
ALEX_NET_PSO_FILE = "alex_pso.h5"
ALEX_NET_GA_FILE = "alex_ga.h5"
VGG_NET_BEST_FILE = "vgg_best.h5"
VGG_NET_PSO_FILE = "vgg_pso.h5"
VGG_NET_GA_FILE = "vgg_ga.h5"
ENSEMBLE_NORMAL_MODEL = "ensemble_normal.h5"

#FILENAME POSITION PSO VARIATION
POS_VAR_LOWER = 'particlesPso.mp4'
POS_VAR_INTER = 'particlesPso_intermedia.mp4'
POS_VAR_HIGHTER = 'particlesPso_elevada.mp4'
POS_VAR_EXP = 'pos_var_exp.html'

# VARIABLES MAKE .mp4 VIDEO with particles movement position
X_LIMITS = [1, 128]
Y_LIMITS = [1, 140]
LABEL_X_AXIS = 'Nºfiltros 1ªcamada'
LABEL_Y_AXIS = 'Nºfiltros 2ªcamada'

# PSO INIT DEFINITIONS --> IN ARGS FORM
pso_init_args_alex = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    7,  # dimensions (nº of normal conv's, nº of stack cnn layers, nº of feature maps of initial conv, growth rate, nº of Dense layers preceding Output Layer, nº neurons of FCL layers (equals along with each other) and batch size)
    np.array(MIN_VALUES_LAYERS_ALEX_NET), # lower bound limits for dimensions
    np.array(MAX_VALUES_LAYERS_ALEX_NET)  # superior bound limits for dimensions
)

pso_init_args_vgg = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    6,  # dimensions (nº of stack cnn layers, nº of feature maps of initial conv, growth rate, nº of Dense layers preceding Output Layer, nº neurons of FCL layers (equals along with each other) and batch size)
    np.array(MIN_VALUES_LAYERS_VGG_NET), # lower bound limits for dimensions
    np.array(MAX_VALUES_LAYERS_VGG_NET)  # superior bound limits for dimensions
)

pso_init_args_resnet = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    5,  # number of filters of first conv layer, number of conv+identity blocks, nº of identity block (all layers - the same value), growth rate and batch size
    np.array(MIN_VALUES_LAYERS_RES_NET),
    np.array(MAX_VALUES_LAYERS_RES_NET)  # superior bound limits for dimensions
)

pso_init_args_densenet = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    6,  # dimensions (init Conv Feature Maps, number of blocks, number cnn layers on blocks, growth rate, comprension rate and batch size)
    np.array(MIN_VALUES_LAYERS_DENSE_NET), # lower bound limits for dimensions
    np.array(MAX_VALUES_LAYERS_DENSE_NET)  # superior bound limits for dimensions
)

## verbose and summary options on build and train
TRAIN_VERBOSE = 1 # 0 - no info, 1- info, 2- partial info
BUILD_SUMMARY = 1 # 0 - no summary, 1- summary