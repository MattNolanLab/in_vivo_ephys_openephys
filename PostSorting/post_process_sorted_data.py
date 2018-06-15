
def process_position_data(recording_to_process, session_type):
    # sync with ephys
    # call functions that are the same

    # call functions different for vr and open field
    if session_type == 'vr':
        pass

    elif session_type == 'openfield':
        pass


def process_spike_data():
    # read firing times and put in array
    pass


def process_light_stimulation():
    pass


def fill_data_frame(spike_data, position_data):
    # calculate scores - hd, grid etc
    pass


def output_cluster_scores():
    pass


def make_plots():
    pass


def post_process_recording(recording_to_process, session_type):
    process_position_data(recording_to_process, session_type)
    process_spike_data()
    process_light_stimulation()
    fill_data_frame()
    output_cluster_scores()
    make_plots()
    pass
