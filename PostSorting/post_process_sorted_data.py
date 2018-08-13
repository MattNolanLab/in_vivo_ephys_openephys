import os
import PostSorting.load_firing_data
import PostSorting.parameters
import PostSorting.open_field_firing_maps
import PostSorting.open_field_firing_fields
import PostSorting.open_field_spatial_data
import PostSorting.open_field_make_plots
import PostSorting.open_field_light_data
import PostSorting.open_field_sync_data
import PostSorting.open_field_spatial_firing
import PostSorting.open_field_head_direction
import PostSorting.temporal_firing
import PostSorting.make_plots

import pandas as pd

prm = PostSorting.parameters.Parameters()


def initialize_parameters(recording_to_process):
    prm.set_is_ubuntu(True)
    prm.set_pixel_ratio(440)
    prm.set_opto_channel('100_ADC3.continuous')
    prm.set_sync_channel('100_ADC1.continuous')
    prm.set_sampling_rate(30000)
    prm.set_local_recording_folder_path(recording_to_process)


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


def process_light_stimulation(recording_to_process, prm):
    opto_on, opto_off, is_found = PostSorting.open_field_light_data.process_opto_data(recording_to_process, prm)  # indices
    return opto_on, opto_off, is_found


def sync_data(recording_to_process, prm, spatial_data):
    synced_spatial_data, is_found = PostSorting.open_field_sync_data.process_sync_data(recording_to_process, prm, spatial_data)
    return synced_spatial_data


def make_plots(position_data, spatial_firing, position_heat_map, hd_histogram, prm):
    PostSorting.make_plots.plot_spike_histogram(spatial_firing, prm)
    PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_spikes_on_trajectory(position_data, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_coverage(position_heat_map, prm)
    PostSorting.open_field_make_plots.plot_firing_rate_maps(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd(spatial_firing, position_data, prm)
    PostSorting.open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, position_data, prm)
    PostSorting.open_field_make_plots.make_combined_figure(prm, spatial_firing)


def create_folders_for_output(recording_to_process):
    if os.path.exists(recording_to_process + '/Figures') is False:
        os.makedirs(recording_to_process + '/Figures')


def save_data_frames(spatial_firing, synced_spatial_data):
    spatial_firing.to_pickle(prm.get_local_recording_folder_path() + '/spatial_firing.pkl')
    synced_spatial_data.to_pickle(prm.get_local_recording_folder_path() + '/position.pkl')


def post_process_recording(recording_to_process, session_type):
    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    spatial_data = process_position_data(recording_to_process, session_type, prm)
    opto_on, opto_off, is_found = process_light_stimulation(recording_to_process, prm)
    synced_spatial_data = sync_data(recording_to_process, prm, spatial_data)
    spike_data = PostSorting.load_firing_data.create_firing_data_frame(recording_to_process, session_type, prm)
    spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, prm)
    spike_data_spatial = PostSorting.open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)
    PostSorting.make_plots.plot_firing_rate_vs_speed(spike_data_spatial, prm)
    hd_histogram, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(spike_data_spatial, synced_spatial_data, prm)
    # PostSorting.open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, prm)

    position_heat_map, spatial_firing = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_spatial, prm)
    spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing)
    save_data_frames(spatial_firing, synced_spatial_data)

    # output_cluster_scores()
    make_plots(synced_spatial_data, spike_data_spatial, position_heat_map, hd_histogram, prm)


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