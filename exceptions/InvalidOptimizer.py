import config

class InvalidOptimizer(Exception):

    def __init__(self, msg = None):

        if msg is None:
            msg = "\nPlease define a valid optimizer: ",config.PSO_OPTIMIZER," or ",config.GA_OPTIMIZER
        super(InvalidOptimizer, self).__init__(msg)