import os
import PostSorting.curation
import PostSorting.load_firing_data
import PostSorting.parameters
import PostSorting.temporal_firing
import PostSorting.vr_spatial_data
import PostSorting.vr_make_plots
import PostSorting.vr_spatial_firing
import PostSorting.vr_firing_maps
import PostSorting.make_plots
import PostSorting.vr_sync_spatial_data
import PostSorting.vr_ramp_cell_test
import PostSorting.vr_firing_maps_copy
import gc

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
    prm.set_file_path(recording_to_process)
    prm.set_local_recording_folder_path(recording_to_process)
    prm.set_output_path(recording_to_process)


def process_position_data(recording_to_process, prm):
    spatial_data = None
    spatial_data = PostSorting.vr_spatial_data.process_position_data(recording_to_process, prm)
    return spatial_data


def make_plots(spike_data, spike_data_movement, spike_data_stationary, raw_position_data, processed_position_data):
    PostSorting.vr_make_plots.plot_stops_on_track(raw_position_data, processed_position_data, prm)
    PostSorting.vr_make_plots.plot_stop_histogram(raw_position_data, processed_position_data, prm)
    PostSorting.vr_make_plots.plot_speed_histogram(processed_position_data, prm)
    gc.collect()
    PostSorting.vr_make_plots.plot_combined_behaviour(raw_position_data, processed_position_data, prm)
    PostSorting.make_plots.plot_waveforms(spike_data, prm)
    PostSorting.make_plots.plot_spike_histogram(spike_data, prm)
    PostSorting.make_plots.plot_autocorrelograms(spike_data, prm)
    gc.collect()
    PostSorting.vr_make_plots.plot_spikes_on_track(spike_data,raw_position_data, processed_position_data, prm, prefix='_all')
    PostSorting.vr_make_plots.plot_spikes_on_track(spike_data_movement,raw_position_data, processed_position_data, prm, prefix='_movement')
    PostSorting.vr_make_plots.plot_spikes_on_track(spike_data_stationary,raw_position_data, processed_position_data, prm, prefix='_stationary')
    gc.collect()
    PostSorting.vr_make_plots.plot_firing_rate_maps(spike_data, prm, prefix='_all')
    PostSorting.vr_make_plots.plot_firing_rate_maps(spike_data_movement, prm, prefix='_movement')
    PostSorting.vr_make_plots.plot_firing_rate_maps(spike_data_stationary, prm, prefix='_stationary')
    PostSorting.vr_make_plots.plot_combined_spike_raster_and_rate(spike_data, raw_position_data, processed_position_data, prm, prefix='_all')
    PostSorting.vr_make_plots.plot_combined_spike_raster_and_rate(spike_data_movement, raw_position_data, processed_position_data, prm, prefix='_movement')
    PostSorting.vr_make_plots.plot_combined_spike_raster_and_rate(spike_data_stationary, raw_position_data, processed_position_data, prm, prefix='_stationary')
    PostSorting.vr_make_plots.make_combined_figure(prm, spike_data, prefix='_all')
    PostSorting.vr_make_plots.make_combined_figure(prm, spike_data, prefix='_movement')
    PostSorting.vr_make_plots.make_combined_figure(prm, spike_data, prefix='_stationary')
    #PostSorting.vr_make_plots.plot_spike_rate_vs_speed(spike_data, processed_position_data, prm)


def save_data_frames(spatial_firing, spatial_data, bad_clusters):
    spatial_firing.to_pickle(prm.get_local_recording_folder_path() + '/DataFrames/spatial_firing.pkl')
    spatial_data.to_pickle(prm.get_local_recording_folder_path() + '/DataFrames/position.pkl')
    bad_clusters.to_pickle(prm.get_local_recording_folder_path() + '/DataFrames/noisy_clusters.pkl')


def create_folders_for_output(recording_to_process):
    if os.path.exists(recording_to_process + '/Figures') is False:
        os.makedirs(recording_to_process + '/Figures')
    if os.path.exists(recording_to_process + '/DataFrames') is False:
        os.makedirs(recording_to_process + '/DataFrames')
    if os.path.exists(recording_to_process + '/Firing_fields') is False:
        os.makedirs(recording_to_process + '/Firing_fields')
    if os.path.exists(recording_to_process + '/Data_test') is False:
        os.makedirs(recording_to_process + '/Data_test')


def post_process_recording(recording_to_process, session_type, sorter_name='MountainSort'):
    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    prm.set_sorter_name('/' + sorter_name)
    prm.set_output_path(recording_to_process + prm.get_sorter_name())
    raw_position_data = PostSorting.vr_sync_spatial_data.syncronise_position_data(recording_to_process, prm)
    raw_position_data, processed_position_data = process_position_data(raw_position_data, prm)

    spike_data = PostSorting.load_firing_data.create_firing_data_frame(recording_to_process, session_type, prm)
    spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, prm)
    spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, prm)
    if len(spike_data) == 0:  # this means that there are no good clusters and the analysis will not run
        PostSorting.vr_make_plots.plot_combined_behaviour(raw_position_data, processed_position_data, prm)
        save_data_frames(spike_data, processed_position_data, bad_clusters)
        return

    spike_data = PostSorting.load_snippet_data.get_snippets(spike_data, prm)
    spike_data, spike_data_movement, spike_data_stationary = PostSorting.vr_spatial_firing.process_spatial_firing(spike_data, raw_position_data)
    #new way of making firing rate maps
    #spike_data = PostSorting.vr_firing_maps_copy.make_firing_field_maps(raw_position_data, spike_data, prm)
    spike_data = PostSorting.vr_firing_maps.make_firing_field_maps(spike_data, raw_position_data, processed_position_data, processed_position_data.binned_time_ms)
    spike_data_movement = PostSorting.vr_firing_maps.make_firing_field_maps(spike_data_movement, raw_position_data, processed_position_data, processed_position_data.binned_time_moving_ms)
    spike_data_stationary = PostSorting.vr_firing_maps.make_firing_field_maps(spike_data_stationary, raw_position_data, processed_position_data, processed_position_data.binned_time_stationary_ms)
    make_plots(spike_data, spike_data_movement, spike_data_stationary, raw_position_data, processed_position_data)
    spike_data = PostSorting.vr_ramp_cell_test.analyse_ramp_firing(prm,spike_data)
    gc.collect()
    save_data_frames(spike_data, processed_position_data, bad_clusters)


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()

    #recording_folder = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D28_2018-09-15_11-49-09'
    recording_folder = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D27_2018-10-05_11-17-55' # test recording
    #recording_folder = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D16_2018-10-11_13-55-54'
    print('Processing ' + str(recording_folder))

    post_process_recording(recording_folder, 'vr')


if __name__ == '__main__':
    main()
