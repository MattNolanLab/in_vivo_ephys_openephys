class Parameters:

    pixel_ratio = None
    opto_channel = ''
    sync_channel = ''
    sampling_rate = 0
    opto_tagging_start_index = None
    sampling_rate_rate = 0

    def __init__(self):
        return

    def get_pixel_ratio(self):
        return Parameters.pixel_ratio

    def set_pixel_ratio(self, pr):
        Parameters.pixel_ratio = pr

    def get_opto_channel(self):
        return Parameters.opto_channel

    def set_opto_channel(self, opto_ch):
        Parameters.opto_channel = opto_ch

    def get_sync_channel(self):
        return Parameters.sync_channel

    def set_sync_channel(self, sync_ch):
        Parameters.sync_channel = sync_ch

    def get_sampling_rate(self):
        return Parameters.sampling_rate

    def set_sampling_rate(self, sr):
        Parameters.sampling_rate = sr

    def get_opto_tagging_start_index(self):
            return Parameters.opto_tagging_start_index

    def set_opto_tagging_start_index(self, opto_start):
        Parameters.opto_tagging_start_index = opto_start

    def get_sampling_rate_rate(self):
            return Parameters.sampling_rate_rate

    def set_sampling_rate_rate(self, sr):
        Parameters.sampling_rate_rate = sr



