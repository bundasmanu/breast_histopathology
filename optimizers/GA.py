from . import Optimizer
from models import Model
from exceptions import NoModel

class GA(Optimizer):

    def __init__(self, model : Model.Model):
        if model == None:
            return NoModel
        self.model = model

    def optimize(self):
        pass