class Parameters:

    is_ubuntu = True
    is_windows = False
    pixel_ratio = None
    opto_channel = ''
    sync_channel = ''
    sampling_rate = 0
    opto_tagging_start_index = None
    sampling_rate_rate = 0
    local_recording_folder_path = ''

    def __init__(self):
        return

    def get_is_ubuntu(self):
        return Parameters.is_ubuntu

    def set_is_ubuntu(self, is_ub):
        Parameters.is_ubuntu = is_ub

    def get_is_windows(self):
        return Parameters.is_windows

    def set_is_windows(self, is_win):
        Parameters.is_windows = is_win

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

    def get_local_recording_folder_path(self):
            return Parameters.local_recording_folder_path

    def set_local_recording_folder_path(self, path):
        Parameters.local_recording_folder_path = path



