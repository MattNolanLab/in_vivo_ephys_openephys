import PostSorting.parameters
import PostSorting.open_field_spatial_data
import PostSorting.open_field_make_plots
import PostSorting.open_field_light_data
import PostSorting.open_field_sync_data

prm = PostSorting.parameters.Parameters()


def initialize_parameters():
    prm.set_pixel_ratio(440)
    prm.set_opto_channel('100_ADC3.continuous')
    prm.set_sync_channel('100_ADC1.continuous')
    prm.set_sampling_rate(30000)


def process_position_data(recording_to_process, session_type, prm):
    spatial_data = None
    # sync with ephys
    # call functions that are the same

    # call functions different for vr and open field
    if session_type == 'vr':
        pass

    elif session_type == 'openfield':
        # dataframe contains time, position coordinates: x, y, head-direction (degrees)
        spatial_data = PostSorting.open_field_spatial_data.process_position_data(recording_to_process, prm)
        # PostSorting.open_field_make_plots.plot_position(spatial_data)

    return spatial_data


def process_spike_data():
    # read firing times and put in array
    pass


def process_light_stimulation(recording_to_process, prm):
    opto_on, opto_off, is_found = PostSorting.open_field_light_data.process_opto_data(recording_to_process, prm)  # indices
    return opto_on, opto_off, is_found


def fill_data_frame(spike_data, position_data):
    # calculate scores - hd, grid etc
    pass


def sync_data(recording_to_process, prm, spatial_data):
    synced_spatial_data, is_found = PostSorting.open_field_sync_data.process_sync_data(recording_to_process, prm, spatial_data)
    return synced_spatial_data


def output_cluster_scores():
    pass


def make_plots():
    pass


def post_process_recording(recording_to_process, session_type):
    initialize_parameters()
    spatial_data = process_position_data(recording_to_process, session_type, prm)
    opto_on, opto_off, is_found = process_light_stimulation(recording_to_process, prm)
    synced_spatial_data = sync_data(recording_to_process, prm, spatial_data)

    process_spike_data()
    sync_data()
    fill_data_frame()
    output_cluster_scores()
    make_plots()
    pass


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()
    params.set_pixel_ratio(440)

    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'
    # recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M13_2018-05-01_11-23-01_of'
    # process_position_data(recording_folder, 'openfield', params)
    post_process_recording(recording_folder, 'openfield')


if __name__ == '__main__':
    main()