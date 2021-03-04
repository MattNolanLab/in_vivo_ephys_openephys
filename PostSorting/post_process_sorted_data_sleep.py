import os
import pickle
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
import PostSorting.speed
import PostSorting.temporal_firing
import PostSorting.open_field_grid_cells
import PostSorting.make_plots
import PostSorting.make_opto_plots
import PostSorting.compare_first_and_second_half
import PostSorting.open_field_border_cells
import PostSorting.theta_modulation
import PostSorting.lfp
import PostSorting.load_snippet_data_opto
import open_ephys_IO

import numpy as np


import pandas as pd

prm = PostSorting.parameters.Parameters()


def initialize_parameters(recording_to_process):
    prm.set_is_ubuntu(True)
    prm.set_pixel_ratio(440)
    prm.set_opto_channel('100_ADC3.continuous')
    if os.path.exists(recording_to_process + '/100_ADC1.continuous'):
        prm.set_sync_channel('100_ADC1.continuous')
    elif os.path.exists(recording_to_process + '/105_CH20_2_0.continuous'):
        prm.set_sync_channel('105_CH20_2_0.continuous')
    else:
        prm.set_sync_channel('105_CH20_0.continuous')

    prm.set_ephys_channels(PostSorting.load_firing_data.available_ephys_channels(recording_to_process, prm))
    prm.set_sampling_rate(30000)
    prm.set_local_recording_folder_path(recording_to_process)
    prm.set_file_path(recording_to_process)  # todo clean this
    prm.set_ms_tmp_path('/tmp/mountainlab/')


def process_running_parameter_tag(running_parameter_tags):
    unexpected_tag = False
    interleaved_opto = False
    delete_first_two_minutes = False
    pixel_ratio = False

    if not running_parameter_tags:
        return unexpected_tag, interleaved_opto, delete_first_two_minutes, pixel_ratio

    tags = [x.strip() for x in running_parameter_tags.split('*')]
    for tag in tags:
        if tag == 'interleaved_opto':
            interleaved_opto = True
        elif tag == 'delete_first_two_minutes':
            delete_first_two_minutes = True
        elif tag.startswith('pixel_ratio'):
            pixel_ratio = int(tag.split('=')[1])  # put pixel ratio value in pixel_ratio
        else:
            print('Unexpected / incorrect tag in the third line of parameters file: ' + str(unexpected_tag))
            unexpected_tag = True
    return unexpected_tag, interleaved_opto, delete_first_two_minutes, pixel_ratio


def process_position_data(recording_to_process, session_type, prm):
    spatial_data = None
    is_found = False
    # dataframe contains time, position coordinates: x, y, head-direction (degrees)
    spatial_data, is_found = PostSorting.open_field_spatial_data.process_position_data(recording_to_process, prm)
    return spatial_data, is_found


def process_light_stimulation(recording_to_process, prm):
    opto_on, opto_off, is_found = PostSorting.open_field_light_data.process_opto_data(recording_to_process, prm)  # indices
    if is_found != None:
        opto_data_frame = PostSorting.open_field_light_data.make_opto_data_frame(opto_on)
        if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
            os.makedirs(prm.get_output_path() + '/DataFrames')
        opto_data_frame.to_pickle(prm.get_output_path() + '/DataFrames/opto_pulses.pkl')
    return opto_on, opto_off, is_found


def sync_data(recording_to_process, prm, spatial_data):
    synced_spatial_data, is_found = PostSorting.open_field_sync_data.process_sync_data(recording_to_process, prm,
                                                                                       spatial_data)
    return synced_spatial_data


def make_plots(position_data, spatial_firing, position_heat_map, prm):
    PostSorting.make_plots.plot_waveforms(spatial_firing, prm)
    PostSorting.make_plots.plot_spike_histogram(spatial_firing, prm)
    PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position_data, prm)
    PostSorting.make_plots.plot_speed_vs_firing_rate(position_data, spatial_firing, prm.get_sampling_rate(), 250, prm)
    PostSorting.make_plots.plot_autocorrelograms(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_spikes_on_trajectory(position_data, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_coverage(position_heat_map, prm)
    PostSorting.make_opto_plots.make_optogenetics_plots(spatial_firing, prm.get_output_path(), prm.get_sampling_rate())
    PostSorting.open_field_make_plots.make_combined_figure(prm, spatial_firing)


def make_plots_with_no_spatial_data(spatial_firing, prm):
    PostSorting.make_plots.plot_waveforms(spatial_firing, prm)
    PostSorting.make_plots.plot_spike_histogram(spatial_firing, prm)
    PostSorting.make_plots.plot_autocorrelograms(spatial_firing, prm)
    PostSorting.make_opto_plots.make_optogenetics_plots(spatial_firing, prm.get_output_path(), prm.get_sampling_rate())
    PostSorting.open_field_make_plots.make_combined_figure(prm, spatial_firing)
    

def create_folders_for_output(recording_to_process):
    if os.path.exists(recording_to_process + '/Figures') is False:
        os.makedirs(recording_to_process + '/Figures')
    if os.path.exists(recording_to_process + '/DataFrames') is False:
        os.makedirs(recording_to_process + '/DataFrames')
    if os.path.exists(recording_to_process + '/Firing_fields') is False:
        os.makedirs(recording_to_process + '/Firing_fields')


def save_data_frames(spatial_firing, synced_spatial_data, snippet_data=None, bad_clusters=None, lfp_data=None):
    print('I will save the data frames now.')
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    spatial_firing.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing.pkl')
    synced_spatial_data.to_pickle(prm.get_output_path() + '/DataFrames/position.pkl')
    if snippet_data is not None:
        snippet_data.to_pickle(prm.get_output_path() + '/DataFrames/snippet_data.pkl')
    if bad_clusters is not None:
        bad_clusters.to_pickle(prm.get_output_path() + '/DataFrames/noisy_clusters.pkl')
    if lfp_data is not None:
        lfp_data.to_pickle(prm.get_output_path() + "/DataFrames/lfp_data.pkl")


def save_data_for_plots(position_heat_map, prm):
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    np.save(prm.get_output_path() + '/DataFrames/position_heat_map.npy', position_heat_map)
    file_handler = open(prm.get_output_path() + '/DataFrames/prm', 'wb')
    pickle.dump(prm, file_handler)


def run_analyses(spike_data_in, synced_spatial_data, opto_analysis=False, lfp_data=None):
    snippet_data = PostSorting.load_snippet_data.get_snippets(spike_data_in, prm, random_snippets=False)
    spike_data = PostSorting.load_snippet_data.get_snippets(spike_data_in, prm, random_snippets=True)
    spike_data_spatial = PostSorting.open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)
    spike_data_spatial = PostSorting.speed.calculate_speed_score(synced_spatial_data, spike_data_spatial, 250,
                                                                 prm.get_sampling_rate())

    position_heat_map, spatial_firing = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data,
                                                                                                  spike_data_spatial,
                                                                                                  prm)
    spatial_firing = PostSorting.theta_modulation.calculate_theta_index(spatial_firing, prm)
    if opto_analysis:
        spatial_firing = PostSorting.load_snippet_data_opto.get_opto_snippets(spatial_firing, prm, random_snippets=True)
        spatial_firing = PostSorting.open_field_light_data.process_spikes_around_light(spike_data_spatial, prm)

    make_plots(synced_spatial_data, spatial_firing, position_heat_map, prm)
    PostSorting.open_field_make_plots.make_combined_field_analysis_figures(prm, spatial_firing)

    save_data_frames(spatial_firing, synced_spatial_data, snippet_data=snippet_data, lfp_data=lfp_data)
    save_data_for_plots(position_heat_map, prm)

    return synced_spatial_data, spatial_firing


def run_analyses_without_position_data(spike_data_in, opto_analysis=False, lfp_data=None):
    snippet_data = PostSorting.load_snippet_data.get_snippets(spike_data_in, prm, random_snippets=False)
    spike_data = PostSorting.load_snippet_data.get_snippets(spike_data_in, prm, random_snippets=True)

    spatial_firing = PostSorting.theta_modulation.calculate_theta_index(spike_data, prm)
    if opto_analysis:
        spatial_firing = PostSorting.load_snippet_data_opto.get_opto_snippets(spatial_firing, prm, random_snippets=True)
        spatial_firing = PostSorting.open_field_light_data.process_spikes_around_light(spike_data, prm)

    make_plots_with_no_spatial_data(spatial_firing, prm)


def set_recording_length(recording_to_process, prm):
    # only use this when there's no position data
    is_found = False
    continuous_channel_data = None
    print('I am loading a channel to find out the length of the recording, because there is no position data available.')
    file_path = recording_to_process + '/' + prm.get_sync_channel()
    if os.path.exists(file_path):
        continuous_channel_data = open_ephys_IO.get_data_continuous(prm, file_path)
        prm.set_total_length_sampling_points(len(continuous_channel_data))
        is_found = True
    else:
        print('I could not load the channel and set the recording length.')
    return continuous_channel_data, is_found


def post_process_recording(recording_to_process, session_type, running_parameter_tags=False, run_type='default',
                           analysis_type='default', sorter_name='MountainSort', stitchpoint=None, paired_order=None, total_length=None):
    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    unexpected_tag, interleaved_opto, delete_first_two_minutes, pixel_ratio = process_running_parameter_tag(
        running_parameter_tags)
    prm.set_stitch_point(stitchpoint)
    prm.set_paired_order(paired_order)
    prm.set_sorter_name('/' + sorter_name)
    prm.set_output_path(recording_to_process + prm.get_sorter_name())
    prm.set_interleaved_opto(interleaved_opto)
    prm.set_delete_two_minutes(delete_first_two_minutes)
    if total_length is not None:
        prm.set_total_length_sampling_points(total_length/prm.get_sampling_rate())
    if pixel_ratio is False:
        print('Default pixel ratio (440) is used.')
    else:
        prm.set_pixel_ratio(pixel_ratio)

    lfp_data = PostSorting.lfp.process_lfp(recording_to_process, prm)
    opto_on, opto_off, opto_is_found = process_light_stimulation(recording_to_process, prm)
    # process spatial data
    position_was_found = False
    try:
        spatial_data, position_was_found = process_position_data(recording_to_process, session_type, prm)
    except:
        print('I cannot analyze the position data for this sleep recording.')

    # analyze spike data
    if not position_was_found:  # this is normally set after syncing the ephys and position data
        set_recording_length(recording_to_process, prm)
    spike_data = PostSorting.load_firing_data.create_firing_data_frame(recording_to_process, session_type, prm)
    spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, prm)
    spike_data = PostSorting.temporal_firing.correct_for_stitch(spike_data, prm)
    spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, prm)
    snippet_data = PostSorting.load_snippet_data.get_snippets(spike_data, prm, random_snippets=False)
    run_analyses_without_position_data(spike_data, opto_analysis=False, lfp_data=None)
    if position_was_found:
        synced_spatial_data = sync_data(recording_to_process, prm, spatial_data)
        if len(spike_data) == 0:  # this means that there are no good clusters and the analysis will not run
            save_data_frames(spike_data, synced_spatial_data, snippet_data=snippet_data, bad_clusters=bad_clusters,lfp_data=lfp_data)
            return

        synced_spatial_data, spatial_firing = run_analyses(spike_data, synced_spatial_data, opto_analysis=opto_is_found, lfp_data=lfp_data)



#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    prm = PostSorting.parameters.Parameters()
    prm.set_pixel_ratio(440)
    prm.set_sampling_rate(30000)
    path_to_recordings = '/home/ubuntu/to_sort/recordings/M5_2018-03-06_15-34-44_of'
    path_to_recordings = 'C:/Users/s1466507/Documents/Ephys/recordings/test_figures'
    prm.set_output_path(path_to_recordings + '/MountainSort/')

    position_data = pd.read_pickle(path_to_recordings + '/MountainSort/DataFrames/position.pkl')
    spatial_firing = pd.read_pickle(path_to_recordings + '/MountainSort/DataFrames/spatial_firing.pkl')
    position_heat_map = np.load(path_to_recordings + '/MountainSort/DataFrames/position_heat_map.npy')
    hd_histogram = np.load(path_to_recordings + '/MountainSort/DataFrames/hd_histogram.npy')

    # filehandler = open('/home/ubuntu/to_sort/recordings/M5_2018-03-06_15-34-44_of/MountainSort/DataFrames/prm', 'rb')
    # prm = pickle.load(filehandler)

    make_plots(position_data, spatial_firing, position_heat_map, hd_histogram, prm)

    # recording_folder = '/home/nolanlab/to_sort/recordings/M5_2018-03-06_15-34-44_of'
    # post_process_recording(recording_folder, 'openfield')


if __name__ == '__main__':
    main()
