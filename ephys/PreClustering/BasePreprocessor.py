import logging

class BasePreproccesor:
    """
    The base class for preprocessor
    """
    def __init__(self, sampling_rate, num_tetrodes, movement_ch,
            opto_ch):
        
        self.logger = logging.getLogger(self.__class__)
        self.sampling_rate = sampling_rate
        self.num_tetrodes = num_tetrodes
        self.movement_ch = movement_ch
        self.opto_ch = opto_ch
    
    def run(self):
        pass

    
    def cleaup(self):
        pass