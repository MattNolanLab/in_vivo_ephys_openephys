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
import PostSorting.open_field_interleaved_opto

import PreClustering.dead_channels

prm = PostSorting.parameters.Parameters()


def create_folders_for_output(recording_to_process):
    if os.path.exists(recording_to_process + '/Figures') is False:
        os.makedirs(recording_to_process + '/Figures')
    if os.path.exists(recording_to_process + '/DataFrames') is False:
        os.makedirs(recording_to_process + '/DataFrames')
    if os.path.exists(recording_to_process + '/Firing_fields') is False:
        os.makedirs(recording_to_process + '/Firing_fields')


def initialize_parameters(recording_to_process):
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
    prm.set_file_path(recording_to_process)
    prm.set_ms_tmp_path('/tmp/mountainlab/')


def process_running_parameter_tag(running_parameter_tags):
    """
    Process tags from parameters.txt metadata file. These are in the third line of the file.
    stimulation_type: continuous or interleaved
    """
    unexpected_tag = False
    pixel_ratio = False
    stimulation_type = False

    if not running_parameter_tags:
        return unexpected_tag, pixel_ratio, stimulation_type

    tags = [x.strip() for x in running_parameter_tags.split('*')]
    for tag in tags:
        if tag.startswith('pixel_ratio'):
            pixel_ratio = int(tag.split('=')[1])  # put pixel ratio value in pixel_ratio
        elif tag.startswith('stimulation_type'):
            stimulation_type = str(tag.split('=')[1])
        else:
            print('Unexpected or missing tag in the third line of parameters file: ' + str(unexpected_tag))
            unexpected_tag = True
    return unexpected_tag, pixel_ratio, stimulation_type


def process_opto_data(recording_to_process, opto_channel):
    opto_on = opto_off = None
    opto_data, is_found = PostSorting.open_field_light_data.load_opto_data(recording_to_process, opto_channel)
    first_opto_pulse_index = None
    last_opto_pulse_index = None
    if is_found:
        opto_on, opto_off = PostSorting.open_field_light_data.get_ons_and_offs(opto_data)
        if not np.asarray(opto_on).size:
            is_found = False
        else:
            first_opto_pulse_index = min(opto_on[0])
            last_opto_pulse_index = max(opto_on[0])
    return opto_on, opto_off, is_found, first_opto_pulse_index, last_opto_pulse_index


def process_light_stimulation(recording_to_process, prm):
    opto_on, opto_off, is_found, opto_start_index, opto_end_index = process_opto_data(recording_to_process, prm.get_opto_channel())
    if is_found:
        opto_data_frame = PostSorting.open_field_light_data.make_opto_data_frame(opto_on)
        if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
            os.makedirs(prm.get_output_path() + '/DataFrames')
        opto_data_frame.to_pickle(prm.get_output_path() + '/DataFrames/opto_pulses.pkl')
    return opto_on, opto_off, is_found, opto_start_index, opto_end_index


def remove_exploration_without_opto(start_of_opto, end_of_opto, spatial_data, sampling_rate):
    # removes data points from before first pulse and after last pulse with 1s buffer on either side
    if start_of_opto is None:
        return spatial_data

    else:
        start_of_opto_seconds = int(start_of_opto / sampling_rate) - 1  # index of first pulse w 1s buffer
        end_of_opto_seconds = int(end_of_opto / sampling_rate) + 1  # index of last pulse w 1s buffer
        # find closest indices for start & end in dataframe
        nearest_bonsai_index_opto_start = (np.abs(spatial_data.synced_time - start_of_opto_seconds)).argmin()
        nearest_bonsai_index_opto_end = (np.abs(spatial_data.synced_time - end_of_opto_seconds)).argmin()
        # drop data on either side of indices
        spatial_data.drop(range(0, nearest_bonsai_index_opto_start), inplace=True)
        spatial_data.drop(range(nearest_bonsai_index_opto_end, spatial_data.index[-1]), inplace=True)
        spatial_data = spatial_data.iloc[:-1, :]  # drop last row
        total_length_seconds = spatial_data.synced_time.values[-1] - spatial_data.synced_time.values[0] # new recording length seconds

    return spatial_data, total_length_seconds


def remove_spikes_without_opto(spike_data, spatial_firing, sampling_rate):
    # remove firing times from before and after start of opto stimulation
    spikes_during_opto = []
    opto_start = int((spatial_firing.synced_time.values[0]) * sampling_rate)  # in sampling points
    opto_end = int((spatial_firing.synced_time.values[-1]) * sampling_rate)  # in sampling points
    recording_length = opto_end - opto_start  # in sampling points

    for cluster_id, cluster in spike_data.iterrows():
        firing_times_all = cluster.firing_times
        firing_times_from_opto_start = np.take(firing_times_all, np.where(firing_times_all >= opto_start)[0])
        firing_times_during_opto = np.take(firing_times_from_opto_start, np.where(firing_times_from_opto_start <= opto_end)[0])
        spikes_during_opto.append(firing_times_during_opto)

    spike_data['firing_times'] = spikes_during_opto  # replace with firing times during opto
    spike_data['firing_times_opto'] = spikes_during_opto  # this col name is required for opto analysis scripts
    spike_data['recording_length_sampling_points'] = int(recording_length)  # replace with opto recording length

    return spike_data


def make_openfield_plots(position_data, spatial_firing, position_heat_map, hd_histogram, output_path, prm):
    PostSorting.make_plots.plot_waveforms(spatial_firing, output_path)
    PostSorting.make_plots.plot_spike_histogram(spatial_firing, output_path)
    PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position_data, prm)
    PostSorting.make_plots.plot_speed_vs_firing_rate(position_data, spatial_firing, prm.get_sampling_rate(), 250, prm)
    PostSorting.make_plots.plot_autocorrelograms(spatial_firing, output_path)
    PostSorting.open_field_make_plots.plot_spikes_on_trajectory(position_data, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_coverage(position_heat_map, prm)
    PostSorting.open_field_make_plots.plot_firing_rate_maps(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_rate_map_autocorrelogram(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd(spatial_firing, position_data, prm)
    PostSorting.open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, position_data, prm)
    PostSorting.open_field_make_plots.plot_spikes_on_firing_fields(spatial_firing, prm)
    PostSorting.open_field_make_plots.make_combined_figure(prm, spatial_firing)


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


def save_data_for_plots(position_heat_map, hd_histogram, prm):
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    np.save(prm.get_output_path() + '/DataFrames/position_heat_map.npy', position_heat_map)
    np.save(prm.get_output_path() + '/DataFrames/hd_histogram.npy', hd_histogram)
    file_handler = open(prm.get_output_path() + '/DataFrames/prm', 'wb')
    pickle.dump(prm, file_handler)


def find_window_size(stimulation_frequency):
    # calculate appropriate size window for plotting - default is 200 ms
    if stimulation_frequency <= 5:
        window_size = 200  # default is appropriate for 5 Hz and below
    else:
        window_size = 1000 / stimulation_frequency

    return int(window_size)


def find_stimulation_frequency(opto_on, sampling_rate):
    # calculate stimulation frequency from time between pulses, return pulse width (ms) and frequency (Hz)
    opto_end_times = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1)[0])
    opto_start_times_from_second = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1)[0] + 1)
    opto_start_times = np.append(opto_on[0][0], opto_start_times_from_second)
    pulse_width_ms = int(((opto_end_times[0] - opto_start_times[0]) / sampling_rate) * 1000)
    between_pulses_ms = int(((opto_start_times[1] - opto_end_times[0]) / sampling_rate) * 1000)
    stimulation_frequency = int(1000/ (pulse_width_ms + between_pulses_ms))
    window_size_for_plots = find_window_size(stimulation_frequency)

    return stimulation_frequency, pulse_width_ms, window_size_for_plots


def save_copy_of_opto_pulses(of_output_path, opto_output_path):
    # saves copy of .pkl containing opto_pulses to OptoAnalysis folder
    # this was written used as an alternative to making changes to the opto-analysis script
    pulses = pd.read_pickle(of_output_path + '/DataFrames/opto_pulses.pkl')
    if os.path.exists(opto_output_path + '/DataFrames') is False:
        os.makedirs(opto_output_path + '/DataFrames')
    pulses.to_pickle(opto_output_path + '/DataFrames/opto_pulses.pkl')


def make_opto_plots(spatial_firing, output_path, prm):
    PostSorting.make_plots.plot_waveforms(spatial_firing, output_path)
    PostSorting.make_plots.plot_spike_histogram(spatial_firing, output_path)
    PostSorting.make_plots.plot_autocorrelograms(spatial_firing, output_path)
    PostSorting.make_opto_plots.make_optogenetics_plots(spatial_firing, output_path, prm.get_sampling_rate())


def post_process_recording(recording_to_process, stimulation_type, running_parameter_tags=False, sorter_name='MountainSort'):
    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    unexpected_tag, pixel_ratio, stimulation_type = process_running_parameter_tag(running_parameter_tags)
    prm.set_sorter_name('/' + sorter_name)
    prm.set_output_path(recording_to_process + prm.get_sorter_name())
    PreClustering.dead_channels.get_dead_channel_ids(prm)
    dead_channels = prm.get_dead_channels()
    ephys_channels = prm.get_ephys_channels()
    output_path = recording_to_process+'/'+settings.sorterName

    if pixel_ratio is False:
        print('Default pixel ratio (440) is used.')
    else:
        prm.set_pixel_ratio(pixel_ratio)

    if stimulation_type is False:
        print('No stimulation type specified. Default stimulation type is continuous.')
        stimulation_type = 'continuous'
    elif stimulation_type == 'continuous':
        print('Stimulation type is', stimulation_type)
    elif stimulation_type == 'interleaved':
        print('Stimulation type is', stimulation_type)
    else:
        print('Unexpected stimulation type specified. Default stimulation type is continuous.')
        stimulation_type = 'continuous'

    # process position and spike data, check for opto data and extract pulses if found
    lfp_data = PostSorting.lfp.process_lfp(recording_to_process, ephys_channels, output_path, dead_channels)
    spatial_data, position_was_found = PostSorting.open_field_spatial_data.process_position_data(recording_to_process, prm, do_resample=False)
    # on and off times for opto pulses,
    opto_on, opto_off, opto_is_found, opto_start_index, opto_end_index = process_light_stimulation(recording_to_process, prm)

    if position_was_found:
        # analyse position and spike data if position data is found
        synced_spatial_data, length_of_recording_sec, is_found = PostSorting.open_field_sync_data.process_sync_data(recording_to_process, prm, spatial_data)
        spike_data = PostSorting.load_firing_data.process_firing_times(recording_to_process, sorter_name, dead_channels)
        # remove position and spike data before and after stimulation period
        synced_spatial_data, length_of_recording_sec = remove_exploration_without_opto(opto_start_index, opto_end_index, synced_spatial_data, prm.sampling_rate)
        spike_data = remove_spikes_without_opto(spike_data, synced_spatial_data, prm.sampling_rate)
        # add temporal firing properties and curate clusters
        spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, length_of_recording_sec)
        spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, sorter_name, prm.get_local_recording_folder_path(), prm.get_ms_tmp_path())

        if len(spike_data) == 0:  # this means that there are no good clusters and the analysis will not run
            save_data_frames(spike_data, synced_spatial_data, snippet_data=None, bad_clusters=bad_clusters, lfp_data=lfp_data)

        else:  # process position data and output as normal for open-field trials
            snippet_data = PostSorting.load_snippet_data.get_snippets(spike_data, recording_to_process, sorter_name, dead_channels, random_snippets=True)
            spike_data_spatial = PostSorting.open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)
            spike_data_spatial = PostSorting.speed.calculate_speed_score(synced_spatial_data, spike_data_spatial, settings.gauss_sd_for_speed_score, settings.sampling_rate)
            hd_histogram, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(spike_data_spatial, synced_spatial_data)
            position_heat_map, spatial_firing = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_spatial)
            spatial_firing = PostSorting.open_field_firing_maps.calculate_spatial_information(spatial_firing, position_heat_map)
            spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)
            spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing, synced_spatial_data, prm)
            spatial_firing = PostSorting.open_field_border_cells.process_border_data(spatial_firing)
            spatial_firing = PostSorting.theta_modulation.calculate_theta_index(spatial_firing, output_path, settings.sampling_rate)
            spatial_firing, spike_data_first, spike_data_second, synced_spatial_data_first, synced_spatial_data_second = \
                PostSorting.compare_first_and_second_half.analyse_half_session_rate_maps(synced_spatial_data, spatial_firing)

            make_openfield_plots(synced_spatial_data, spatial_firing, position_heat_map, hd_histogram, output_path, prm)
            PostSorting.open_field_make_plots.make_combined_field_analysis_figures(prm, spatial_firing)
            save_data_frames(spatial_firing, synced_spatial_data, snippet_data=None, lfp_data=lfp_data)
            save_data_for_plots(position_heat_map, hd_histogram, prm)

    # analyse opto data, if it was found
    if opto_is_found:
        prm.set_output_path(output_path + '/Opto')  # set new output folder for peristimulus spike analysis
        output_path_opto = prm.get_output_path()
        save_copy_of_opto_pulses(output_path, output_path_opto)  # save copy of opto_pulses.pkl in new folder
        try:
            frequency, pulse_width_ms, window_ms = find_stimulation_frequency(opto_on, prm.sampling_rate)
            print('Stimulation frequency is', frequency, 'where each pulse is', pulse_width_ms, 'ms')
            print('I will use a window of', window_ms, 'ms')
        except:
            print('Stimulation frequency cannot be determined.')
            print('Default window size of 200 ms will be used. This is not appropriate for stimulation frequencies > 5 Hz.')

        print('I will now process the peristimulus spikes. This will take a while for high frequency stimulations in the open-field.')
        spatial_firing = PostSorting.open_field_light_data.process_spikes_around_light(spatial_firing, prm, window_size_ms=window_ms)
        make_opto_plots(spatial_firing, output_path_opto, prm)

        if stimulation_type == 'interleaved':
            print('I will now call scripts to process interleaved opto stimulation.')
            PostSorting.open_field_interleaved_opto.analyse_interleaved_stimulation(spatial_firing, opto_on, frequency)
