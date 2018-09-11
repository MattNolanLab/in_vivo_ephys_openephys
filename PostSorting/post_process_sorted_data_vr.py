import os
import PostSorting.load_firing_data
import PostSorting.parameters
import PostSorting.vr_spatial_data
import PostSorting.vr_make_plots
import PostSorting.vr_spatial_firing
import PostSorting.vr_firing_maps
import PostSorting.density_estimation_sskernel

prm = PostSorting.parameters.Parameters()


def initialize_parameters(recording_to_process):
    prm.set_is_ubuntu(True)
    prm.set_sampling_rate(30000)
    prm.set_local_recording_folder_path(recording_to_process)
    prm.set_opto_channel('100_ADC3.continuous')

    prm.set_stop_threshold(0.7)  # speed is given in cm/200ms 0.7*1/2000
    prm.set_movement_channel('100_ADC2.continuous')
    prm.set_first_trial_channel('100_ADC4.continuous')
    prm.set_second_trial_channel('100_ADC5.continuous')


def process_position_data(recording_to_process):
    vr_spatial_data = None
    vr_spatial_data = PostSorting.vr_spatial_data.process_position_data(recording_to_process)
    return vr_spatial_data


def make_plots(spike_data, spatial_data):
    #PostSorting.vr_make_plots.plot_stops_on_track(spatial_data)
    #PostSorting.vr_make_plots.plot_spikes_on_track(spike_data,spatial_data)
    PostSorting.vr_make_plots.plot_firing_rate_maps(spike_data)


def create_folders_for_output(recording_to_process):
    if os.path.exists(recording_to_process + '/Figures') is False:
        os.makedirs(recording_to_process + '/Figures')


def post_process_recording(recording_to_process, session_type):
    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    spatial_data = process_position_data(recording_to_process)
    spike_data = PostSorting.load_firing_data.create_firing_data_frame(recording_to_process, session_type, prm)

    spike_data = PostSorting.vr_spatial_firing.process_spatial_firing(spike_data, spatial_data)
    spike_data = PostSorting.vr_firing_maps.make_firing_field_maps(spike_data, spatial_data)

    make_plots(spike_data, spatial_data)


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()

    recording_folder = '/Users/sarahtennant/Work/Analysis/Opto_data/mcos/M1_D16_2018-06-21_13-38-50'
    print('Processing ' + str(recording_folder))

    post_process_recording(recording_folder, 'vr')


if __name__ == '__main__':
    main()
