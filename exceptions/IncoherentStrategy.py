
class IncoherentStrategy(Exception):

    def __init__(self, msg=None):
        if msg is None:
            msg = "\nYou cannot choose the oversampling and undersampling strategies at the same time"
        super(IncoherentStrategy, self).__init__(msg)