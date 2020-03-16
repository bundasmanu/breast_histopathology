from . import Optimizer
from models import Model
from exceptions import CustomError
import config

class PSO(Optimizer.Optimizer):

    def __init__(self):
        super(PSO, self).__init__()

    def optimize(self):
        pass