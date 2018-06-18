import PostSorting.parameters
import PostSorting.open_field_spatial_data
import PostSorting.open_field_make_plots

prm = PostSorting.parameters.Parameters()


def initialize_parameters():
    prm.set_pixel_ratio(440)


def process_position_data(recording_to_process, session_type, prm):
    # sync with ephys
    # call functions that are the same

    # call functions different for vr and open field
    if session_type == 'vr':
        pass

    elif session_type == 'openfield':
        # dataframe contains time, position coordinates: x, y, head-direction (degrees)
        spatial_data = PostSorting.open_field_spatial_data.process_position_data(recording_to_process, prm)
        # PostSorting.open_field_make_plots.plot_position(spatial_data)



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
    initialize_parameters()
    process_position_data(recording_to_process, session_type, prm)
    process_spike_data()
    process_light_stimulation()
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
    # recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M0_2017-11-21_15-52-53'
    process_position_data(recording_folder, 'openfield', params)


if __name__ == '__main__':
    main()