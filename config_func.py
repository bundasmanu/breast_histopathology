import cv2
import numpy
import config

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

        for i in len(data):
            image = cv2.imread(data.at[i, config.IMAGE_PATH])
            x.append(cv2.resize(image, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_CUBIC))
            y.append(data.at[i, config.TARGET])

        return numpy.array(x), numpy.array(y)

    except:
        raise