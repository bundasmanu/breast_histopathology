from . import Optimizer
from models import Model
from exceptions import CustomError
import config

class GA(Optimizer):

    def __init__(self, model : Model.Model, *args):
        if model == None:
            return CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)
        self.model = model

    def optimize(self):
        pass