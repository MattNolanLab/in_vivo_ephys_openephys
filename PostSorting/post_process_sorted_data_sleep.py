import os
import pickle

import numpy as np
import pandas as pd
import settings

import PostSorting.compare_first_and_second_half
import PostSorting.curation
import PostSorting.lfp
import PostSorting.load_firing_data
import PostSorting.load_snippet_data
import PostSorting.make_opto_plots
import PostSorting.make_plots
import PostSorting.open_field_border_cells
import PostSorting.open_field_firing_fields
import PostSorting.open_field_firing_maps
import PostSorting.open_field_grid_cells
import PostSorting.open_field_head_direction
import PostSorting.open_field_light_data
import PostSorting.open_field_make_plots
import PostSorting.open_field_spatial_data
import PostSorting.open_field_spatial_firing
import PostSorting.open_field_sync_data
import PostSorting.parameters
import PostSorting.speed
import PostSorting.temporal_firing
import PostSorting.theta_modulation
import PostSorting.load_snippet_data_opto


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
    """
    Process tags from parameters.txt metadata file. These are in the third line of the file.
    """
    unexpected_tag = False
    pixel_ratio = False

    if not running_parameter_tags:
        return unexpected_tag, pixel_ratio

    tags = [x.strip() for x in running_parameter_tags.split('*')]
    for tag in tags:
        if tag.startswith('pixel_ratio'):
            pixel_ratio = int(tag.split('=')[1])  # put pixel ratio value in pixel_ratio
        else:
            print('Unexpected / incorrect tag in the third line of parameters file: ' + str(unexpected_tag))
            unexpected_tag = True
    return unexpected_tag, pixel_ratio


def process_position_data(recording_to_process, session_type, prm, do_resample=False):
    spatial_data = None
    is_found = False
    if session_type == 'openfield':
        # dataframe contains time, position coordinates: x, y, head-direction (degrees)
        spatial_data, is_found = PostSorting.open_field_spatial_data.process_position_data(recording_to_process,prm, do_resample)
        # PostSorting.open_field_make_plots.plot_position(spatial_data)
    return spatial_data, is_found


def process_light_stimulation(recording_to_process, prm):
    opto_on, opto_off, is_found, opto_start_index = PostSorting.open_field_light_data.process_opto_data(recording_to_process, prm)  # indices
    if is_found != None:
        opto_data_frame = PostSorting.open_field_light_data.make_opto_data_frame(opto_on)
        if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
            os.makedirs(prm.get_output_path() + '/DataFrames')
        opto_data_frame.to_pickle(prm.get_output_path() + '/DataFrames/opto_pulses.pkl')
    return opto_on, opto_off, is_found, opto_start_index


def sync_data(recording_to_process, prm, spatial_data):
    synced_spatial_data,total_length_sampling_points, is_found = PostSorting.open_field_sync_data.process_sync_data(recording_to_process, prm,
                                                                                       spatial_data)
    return synced_spatial_data, total_length_sampling_points


def make_plots(position_data, spatial_firing, output_path, prm):
    PostSorting.make_plots.plot_waveforms(spatial_firing, output_path)
    PostSorting.make_plots.plot_spike_histogram(spatial_firing, output_path)
    PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position_data, prm)
    PostSorting.make_plots.plot_autocorrelograms(spatial_firing, output_path)
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


def save_data_for_plots(hd_histogram, prm):
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    np.save(prm.get_output_path() + '/DataFrames/hd_histogram.npy', hd_histogram)
    file_handler = open(prm.get_output_path() + '/DataFrames/prm', 'wb')
    pickle.dump(prm, file_handler)


def post_process_recording(recording_to_process, session_type, total_length=False, running_parameter_tags=False,
                           sorter_name='MountainSort', stitchpoint=None, paired_order=None):
    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    unexpected_tag, pixel_ratio = process_running_parameter_tag(running_parameter_tags)
    prm.set_stitch_point(stitchpoint)
    prm.set_paired_order(paired_order)
    prm.set_sorter_name('/' + sorter_name)
    prm.set_output_path(recording_to_process + prm.get_sorter_name())

    dead_channels = prm.get_dead_channels()
    ephys_channels = prm.get_ephys_channels()
    output_path = recording_to_process+'/'+settings.sorterName

    if pixel_ratio is False:
        print('Default pixel ratio (440) is used.')
    else:
        prm.set_pixel_ratio(pixel_ratio)

    lfp_data = PostSorting.lfp.process_lfp(recording_to_process, ephys_channels, output_path, dead_channels)
    opto_on, opto_off, opto_is_found, opto_start_index = process_light_stimulation(recording_to_process, prm)
    # process spatial data
    spatial_data, position_was_found = process_position_data(recording_to_process, session_type, prm)
    if position_was_found:
        synced_spatial_data, total_length_sampling_points = sync_data(recording_to_process, prm, spatial_data)
        if not total_length:
            total_length = total_length_sampling_points
        # analyze spike data
        spike_data = PostSorting.load_firing_data.create_firing_data_frame(recording_to_process, sorter_name, dead_channels, paired_order, stitchpoint, opto_tagging_start_index=opto_start_index)
        spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, stitchpoint, paired_order, total_length)
        spike_data = PostSorting.temporal_firing.correct_for_stitch(spike_data, paired_order, stitchpoint)
        spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, sorter_name, prm.get_local_recording_folder_path(), prm.get_ms_tmp_path())
        snippet_data = PostSorting.load_snippet_data.get_snippets(spike_data, recording_to_process, sorter_name, dead_channels, random_snippets=False)

        if len(spike_data) == 0:  # this means that there are no good clusters and the analysis will not run
            save_data_frames(spike_data, synced_spatial_data, snippet_data=snippet_data, bad_clusters=bad_clusters,lfp_data=lfp_data)

        else:
            snippet_data = PostSorting.load_snippet_data.get_snippets(spike_data, recording_to_process, sorter_name, dead_channels, random_snippets=True)
            spatial_firing = PostSorting.open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)
            spatial_firing = PostSorting.speed.calculate_speed_score(synced_spatial_data, spatial_firing, settings.gauss_sd_for_speed_score, settings.sampling_rate)

            if opto_is_found:
                spatial_firing = PostSorting.open_field_light_data.process_spikes_around_light(spatial_firing, prm)

            make_plots(synced_spatial_data, spatial_firing, output_path, prm)

            save_data_frames(spatial_firing, synced_spatial_data, snippet_data=snippet_data, lfp_data=lfp_data)




