
class NoModel(Exception):

    def __init__(self, msg=None):

        if msg is None:
            msg = "\nPlease pass a initialized model"
        super(NoModel, self).__init__(msg)