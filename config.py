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

SIZE_DATAFRAME = 10000

VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2

ALEX_NET = "ALEXNET"
VGG_NET = "VGGNET"

PSO_OPTIMIZER = "PSO"
GA_OPTIMIZER = "GA"

RELU_FUNCTION = "relu"
SOFTMAX_FUNCTION = "softmax"

VALID_PADDING = "valid"
SAME_PADDING = "same"