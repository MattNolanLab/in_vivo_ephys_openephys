import os
import PostSorting.curation
import PostSorting.load_firing_data
import PostSorting.load_snippet_data
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
import PostSorting.open_field_grid_cells
import PostSorting.make_plots
import PostSorting.compare_first_and_second_half

import matplotlib.pylab as plt

import pandas as pd

prm = PostSorting.parameters.Parameters()


def initialize_parameters(recording_to_process):
    prm.set_is_ubuntu(True)
    prm.set_pixel_ratio(440)
    prm.set_opto_channel('100_ADC3.continuous')
    prm.set_sync_channel('100_ADC1.continuous')
    prm.set_sampling_rate(30000)
    prm.set_local_recording_folder_path(recording_to_process)
    prm.set_file_path(recording_to_process)  # todo clean this


def process_position_data(recording_to_process, session_type, prm):
    spatial_data = None
    is_found = False
    # sync with ephys
    # call functions that are the same

    # call functions different for vr and open field
    if session_type == 'vr':
        pass

    elif session_type == 'openfield':
        # dataframe contains time, position coordinates: x, y, head-direction (degrees)
        spatial_data, is_found = PostSorting.open_field_spatial_data.process_position_data(recording_to_process, prm)
        # PostSorting.open_field_make_plots.plot_position(spatial_data)

    return spatial_data, is_found


def process_light_stimulation(recording_to_process, prm):
    opto_on, opto_off, is_found = PostSorting.open_field_light_data.process_opto_data(recording_to_process, prm)  # indices
    return opto_on, opto_off, is_found


def sync_data(recording_to_process, prm, spatial_data):
    synced_spatial_data, is_found = PostSorting.open_field_sync_data.process_sync_data(recording_to_process, prm, spatial_data)
    return synced_spatial_data


def make_plots(position_data, spatial_firing, position_heat_map, hd_histogram, prm):
    PostSorting.make_plots.plot_waveforms(spatial_firing, prm)
    PostSorting.make_plots.plot_spike_histogram(spatial_firing, prm)
    PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position_data, prm)
    PostSorting.make_plots.plot_autocorrelograms(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_spikes_on_trajectory(position_data, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_coverage(position_heat_map, prm)
    PostSorting.open_field_make_plots.plot_firing_rate_maps(spatial_firing, prm)
    # PostSorting.open_field_make_plots.plot_rate_map_autocorrelogram(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd(spatial_firing, position_data, prm)
    PostSorting.open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, position_data, prm)
    PostSorting.open_field_make_plots.plot_spikes_on_firing_fields(spatial_firing, prm)
    PostSorting.open_field_make_plots.make_combined_figure(prm, spatial_firing)


def create_folders_for_output(recording_to_process):
    if os.path.exists(recording_to_process + '/Figures') is False:
        os.makedirs(recording_to_process + '/Figures')
    if os.path.exists(recording_to_process + '/DataFrames') is False:
        os.makedirs(recording_to_process + '/DataFrames')
    if os.path.exists(recording_to_process + '/Firing_fields') is False:
        os.makedirs(recording_to_process + '/Firing_fields')


def save_data_frames(spatial_firing, synced_spatial_data, bad_clusters=None):
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    spatial_firing.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing.pkl')
    synced_spatial_data.to_pickle(prm.get_output_path() + '/DataFrames/position.pkl')
    if bad_clusters is not None:
        bad_clusters.to_pickle(prm.get_output_path() + '/DataFrames/noisy_clusters.pkl')


#  this only calls stable analysis functions
def call_stable_functions(recording_to_process, session_type, analysis_type):
    # process opto data -this has to be done before splitting the session into recording and opto-tagging parts
    opto_on, opto_off, is_found = process_light_stimulation(recording_to_process, prm)
    # process spatial data
    spatial_data, position_was_found = process_position_data(recording_to_process, session_type, prm)
    if position_was_found:
        synced_spatial_data = sync_data(recording_to_process, prm, spatial_data)
        # analyze spike data
        spike_data = PostSorting.load_firing_data.create_firing_data_frame(recording_to_process, session_type, prm)
        spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, prm)
        if analysis_type is 'default':
            spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, prm)
            if len(spike_data) == 0:  # this means that there are no good clusters and the analysis will not run
                save_data_frames(spike_data, synced_spatial_data, bad_clusters=bad_clusters)
                return
        spike_data = PostSorting.load_snippet_data.get_snippets(spike_data, prm)
        spike_data_spatial = PostSorting.open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)
        hd_histogram, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(spike_data_spatial, synced_spatial_data, prm)
        position_heat_map, spatial_firing = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_spatial, prm)
        # spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)
        # spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing, synced_spatial_data, prm)
        save_data_frames(spatial_firing, synced_spatial_data)
        make_plots(synced_spatial_data, spatial_firing, position_heat_map, hd_histogram, prm)


def run_analyses(spike_data_in, synced_spatial_data):
    spike_data = PostSorting.load_snippet_data.get_snippets(spike_data_in, prm)
    spike_data_spatial = PostSorting.open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)
    hd_histogram, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(spike_data_spatial, synced_spatial_data, prm)
    position_heat_map, spatial_firing = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_spatial, prm)
    # spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)
    spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing, synced_spatial_data, prm)
    save_data_frames(spatial_firing, synced_spatial_data)
    make_plots(synced_spatial_data, spatial_firing, position_heat_map, hd_histogram, prm)
    return synced_spatial_data, spatial_firing


def post_process_recording(recording_to_process, session_type, run_type='default', analysis_type='default', sorter_name='MS'):
    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    prm.set_sorter_name('/' + sorter_name)
    prm.set_output_path(recording_to_process + prm.get_sorter_name())

    if run_type == 'stable':
        prm.set_is_stable(True)
        call_stable_functions(recording_to_process, session_type, analysis_type)

    if run_type == 'default':
        # process opto data -this has to be done before splitting the session into recording and opto-tagging parts
        opto_on, opto_off, is_found = process_light_stimulation(recording_to_process, prm)
        # process spatial data
        spatial_data, position_was_found = process_position_data(recording_to_process, session_type, prm)
        if position_was_found:
            synced_spatial_data = sync_data(recording_to_process, prm, spatial_data)
            # analyze spike data
            spike_data = PostSorting.load_firing_data.create_firing_data_frame(recording_to_process, session_type, prm)
            spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, prm)
            if analysis_type is 'default':
                spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, prm)
                if len(spike_data) == 0:  # this means that there are no good clusters and the analysis will not run
                    save_data_frames(spike_data, synced_spatial_data, bad_clusters)
                    return
            synced_spatial_data, spatial_firing = run_analyses(spike_data, synced_spatial_data)
            spike_data = PostSorting.compare_first_and_second_half.analyse_first_and_second_halves(prm, synced_spatial_data, spatial_firing)
            save_data_frames(spike_data, synced_spatial_data)



#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()
    params.set_pixel_ratio(440)

    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'
    # recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M13_2018-05-01_11-23-01_of'
    # process_position_data(recording_folder, 'openfield', params)
    #post_process_recording(recording_folder, 'openfield', run_type='stable', analysis_type='get_noisy_clusters', sorter_name='MS')
    # post_process_recording(recording_folder, 'openfield', run_type='stable', analysis_type='default')
    post_process_recording(recording_folder, 'openfield')



if __name__ == '__main__':
    main()