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

SIZE_DATAFRAME = 180000

VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2

ALEX_NET = "ALEXNET"
VGG_NET = "VGGNET"

PSO_OPTIMIZER = "PSO"
GA_OPTIMIZER = "GA"

RELU_FUNCTION = "relu"
SOFTMAX_FUNCTION = "softmax"

LEARNING_RATE = 0.001
DECAY = 1e-6

LOSS_BINARY = "binary_crossentropy"
LOSS_CATEGORICAL = "categorical_crossentropy"

VALID_PADDING = "valid"
SAME_PADDING = "same"

ACCURACY_METRIC = "accuracy"
VALIDATION_ACCURACY = "val_accuracy"

BATCH_SIZE_ALEX_NO_AUG = 180
BATCH_SIZE_ALEX_AUG = 128
EPOCHS = 10
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

#PSO OPTIONS
PARTICLES = 2
ITERATIONS = 2
PSO_DIMENSIONS = 6
TOPOLOGY_FLAG = 0 # 0 MEANS GBEST, AND 1 MEANS LBEST
gbestOptions = {'w' : 0.9, 'c1' : 0.3, 'c2' : 0.3}
lbestOptions = {'w' : 0.9, 'c1' : 0.3, 'c2' : 0.3, 'k' : 4, 'p' : 2}

MAX_VALUES_LAYERS_ALEX_NET = [16, 32, 64, 64, 32, 16]

IDC_CLASS_NAME = "With IDC"
HEALTHY_CLASS_NAME = "Healthy"
LIST_CLASSES_NAME = [HEALTHY_CLASS_NAME, IDC_CLASS_NAME]

#GA OPTIONS
TOURNAMENT_SIZE = 100
INDPB = 0.6
CXPB = 0.4
MUTPB = 0.2