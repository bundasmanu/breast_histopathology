from . import Optimizer
from models import Model
from exceptions import CustomError
import config

class GA(Optimizer.Optimizer):

    def __init__(self, model : Model.Model, *args):
        super(GA, self).__init__(model, *args)

    def optimize(self):
        pass