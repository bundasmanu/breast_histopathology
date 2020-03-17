from . import Optimizer
from models import Model
from exceptions import CustomError
import config
import pyswarms as ps
import numpy as np
from typing import Tuple
import Data

class PSO(Optimizer.Optimizer):

    def __init__(self, model : Model.Model, *args):
        super(PSO, self).__init__(model, *args)

    def boundsDefinition(self):

        '''
        This function has as main objective to define the limits of the dimensions of the problem
        :return: 2 numpy arrays --> shape(dimensionsProblem, ) with min and max values for each dimension of the problem
        '''

        try:

            totalDimensions = self.dims

            minBounds = np.ones(totalDimensions)
            maxBounds = np.ones(totalDimensions)

            maxBounds = [maxBounds[i]*i for i in config.MAX_VALUES_LAYERS_ALEX_NET]
            maxBounds = np.array(maxBounds)

            bounds = (minBounds, maxBounds)

            return bounds

        except:
            raise
    
    def objectiveFunction(self, score, *args):
        return super(PSO, self).objectiveFunction(score, *args)

    def loopAllParticles(self, particles, data : Data.Data):

        '''
        THIS FUNCTION APPLIES PARTICLES ITERATION, EXECUTION CNN MODEL
        :param particles: numpy array of shape (nParticles, dimensions)
        :param X_train: numpy array --> training data
        :param X_val: numpy array --> validation data
        :param X_test: numpy array --> test data
        :param y_train: numpy array --> targets of training data
        :param y_val: numpy array --> targets of validation data
        :param y_test: numpy array --> targets of test data
        :return: list: all losses returned along all particles iteration
        '''

        try:

            losses = []
            for i in range(particles.shape[0]):
                model, predictions, history = self.model.template_method(
                    data=data, *particles[i]
                )
                acc = (data.y_test == predictions).mean()
                losses.append(self.objectiveFunction(acc, *particles[i]))
            return losses

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OPTIMIZATION)

    def optimize(self, data : Data.Data) -> Tuple[float, float]:

        '''
        THIS FUNCTION IS RESPONSIBLE TO APPLY ALL LOGIC OF PSO CNN NETWORK OPTIMIZATION
        :param X_train: numpy array --> training data
        :param X_val: numpy array --> validation data
        :param X_test: numpy array --> test data
        :param y_train: numpy array --> targets of training data
        :param y_val: numpy array --> targets of validation data
        :param y_test: numpy array --> targets of test data
        :return: [float, float] --> best cost and best particle position
        '''

        try:

            #DEFINITION OF BOUNDS
            bounds = self.boundsDefinition()

            optimizer = None
            if config.TOPOLOGY_FLAG == 0: #global best topology
                optimizer = ps.single.GlobalBestPSO(n_particles=self.indiv, dimensions=self.dims,
                                                    options=config.gbestOptions, bounds=bounds)
            else: #local best topology
                optimizer = ps.single.GlobalBestPSO(n_particles=self.indiv, dimensions=self.dims,
                                                    options=config.lbestOptions, bounds=bounds)

            cost, pos = optimizer.optimize(objective_func=self.loopAllParticles, data=data , iters=self.iters)

            return cost, pos

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OPTIMIZATION)