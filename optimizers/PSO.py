from . import Optimizer
from models import Model
from exceptions import CustomError
import config
import pyswarms as ps

class PSO(Optimizer.Optimizer):

    def __init__(self, model : Model.Model):
        super(PSO, self).__init__(model)

    def optimize(self):

        try:

            #DEFINITION OF BOUNDS
            # self.
            #
            # if config.TOPOLOGY_FLAG == 0: #global best topology
            #     optimizer = ps.single.GlobalBestPSO(n_particles=config.PARTICLES, dimensions=config.DIMENSIONS,
            #                                         options=config.gbestOptions, bounds=)

            return None

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OPTIMIZATION)