from abc import ABC, abstractmethod
from models import Model
from exceptions import CustomError
import config

class Optimizer(ABC):

    def __init__(self, model : Model.Model):
        if model == None:
            return CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)
        self.model = model

    @abstractmethod
    def optimize(self):
        pass