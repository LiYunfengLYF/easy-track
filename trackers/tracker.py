class Tracker:
    def __init__(self):
        pass

    def init(self, image, info):
        raise NotImplementedError

    def track(self, image):
        raise NotImplementedError
