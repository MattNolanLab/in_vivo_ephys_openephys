from BasePreprocessor import BasePreproccesor


class MountainSortPreprocessor(BasePreproccesor):
    def __init__(self,sampling_rate = 30000, num_tetrodes=4, movement_ch='100_ADC2.continuous',
            opto_ch='100_ADC3.continuous'):
        super().__init__(sampling_rate,num_tetrodes,movement_ch,opto_ch)

    
    def run(self):
        pass

    
    def cleanup(self):
        pass

    