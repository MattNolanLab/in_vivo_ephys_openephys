import logging

class BaseSorter:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        pass

    def run(self):
        pass

    def cleanup(self):
        pass