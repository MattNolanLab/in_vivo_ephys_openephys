import os
import PostSorting.curation
import PostSorting.load_firing_data
import PostSorting.load_snippet_data
import PostSorting.parameters
import PostSorting.temporal_firing
import PostSorting.vr_spatial_data
import PostSorting.vr_make_plots
import PostSorting.vr_spatial_firing
import PostSorting.make_plots
import PostSorting.vr_sync_spatial_data
import PostSorting.vr_firing_rate_maps
import PostSorting.vr_FiringMaps_InTime
import gc
from tqdm import tqdm
import pandas as pd
import PostSorting.vr_cued

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
    prm.set_goal_location_chennl('100_ADC7.continuous')
    prm.set_file_path(recording_to_process)
    prm.set_local_recording_folder_path(recording_to_process)
    prm.set_ms_tmp_path('/tmp/mountainlab/')


def process_position_data(recording_to_process, prm):
    raw_position_data = PostSorting.vr_sync_spatial_data.syncronise_position_data(recording_to_process, prm)
    raw_position_data, processed_position_data = PostSorting.vr_spatial_data.process_position(raw_position_data, prm, recording_to_process)
    return raw_position_data, processed_position_data


def process_firing_properties(recording_to_process, session_type, prm):
    spike_data = PostSorting.load_firing_data.create_firing_data_frame(recording_to_process, session_type, prm)
    spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, prm)
    spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, prm)
    return spike_data, bad_clusters


def make_plots(spike_data, raw_position_data, processed_position_data):
    PostSorting.vr_make_plots.plot_stops_on_track(raw_position_data, processed_position_data, prm)
    PostSorting.vr_make_plots.plot_stop_histogram(raw_position_data, processed_position_data, prm)
    PostSorting.vr_make_plots.plot_speed_histogram(raw_position_data, processed_position_data, prm)
    PostSorting.make_plots.plot_waveforms(spike_data, prm)
    PostSorting.make_plots.plot_spike_histogram(spike_data, prm)
    PostSorting.make_plots.plot_autocorrelograms(spike_data, prm)
    gc.collect()
    PostSorting.vr_make_plots.plot_spikes_on_track(spike_data,raw_position_data, processed_position_data, prm, prefix='_movement')
    gc.collect()
    PostSorting.vr_make_plots.plot_firing_rate_maps(spike_data, prm, prefix='_all')
    PostSorting.vr_make_plots.plot_convolved_rates_in_time(spike_data, prm)
    #PostSorting.vr_make_plots.plot_combined_spike_raster_and_rate(spike_data, raw_position_data, processed_position_data, prm, prefix='_all')
    #PostSorting.vr_make_plots.make_combined_figure(prm, spike_data, prefix='_all')


def save_data_frames(prm, spatial_firing_movement, spatial_firing_stationary, spatial_firing, raw_position_data, processed_position_data, bad_clusters):
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    spatial_firing_movement.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing.pkl')
    spatial_firing_stationary.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing_stationary.pkl')
    spatial_firing.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing_all.pkl')
    raw_position_data.to_pickle(prm.get_output_path() + '/DataFrames/raw_position_data.pkl')
    processed_position_data.to_pickle(prm.get_output_path() + '/DataFrames/processed_position_data.pkl')
    bad_clusters.to_pickle(prm.get_output_path() + '/DataFrames/noisy_clusters.pkl')
    snippet_data.to_pickle(prm.get_output_path() + '/DataFrames/snippet_data.pkl')


def save_noisy_cluster_frames(prm, spatial_firing, raw_position_data, processed_position_data, bad_clusters):
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    spatial_firing.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing_all.pkl')
    raw_position_data.to_pickle(prm.get_output_path() + '/DataFrames/raw_position_data.pkl')
    processed_position_data.to_pickle(prm.get_output_path() + '/DataFrames/processed_position_data.pkl')
    bad_clusters.to_pickle(prm.get_output_path() + '/DataFrames/noisy_clusters.pkl')



def create_folders_for_output(recording_to_process):
    if os.path.exists(recording_to_process + '/Figures') is False:
        os.makedirs(recording_to_process + '/Figures')
    if os.path.exists(recording_to_process + '/DataFrames') is False:
        os.makedirs(recording_to_process + '/DataFrames')
    if os.path.exists(recording_to_process + '/Firing_fields') is False:
        os.makedirs(recording_to_process + '/Firing_fields')
    if os.path.exists(recording_to_process + '/Data_test') is False:
        os.makedirs(recording_to_process + '/Data_test')

def process_running_parameter_tag(running_parameter_tags):
    stop_threshold = 4.9  # defaults
    track_length = 200 # default assumptions
    cue_conditioned_goal = False

    if not running_parameter_tags:
        return stop_threshold, track_length, cue_conditioned_goal

    tags = [x.strip() for x in running_parameter_tags.split('*')]
    for tag in tags:
        if tag.startswith('stop_threshold'):
            stop_threshold = float(tag.split("=")[1])
        elif tag.startswith('track_length'):
            track_length = int(tag.split("=")[1])
        elif tag.startswith('cue_conditioned_goal'):
            cue_conditioned_goal = bool(tag.split('=')[1])
        else:
            print('Unexpected / incorrect tag in the third line of parameters file: ' + str(unexpected_tag))
            unexpected_tag = True
    return stop_threshold, track_length, cue_conditioned_goal


def post_process_recording(recording_to_process, session_type, running_parameter_tags=False, sorter_name='MountainSort'):
    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    stop_threshold, track_length, cue_conditioned_goal = process_running_parameter_tag(running_parameter_tags)
    prm.set_stop_threshold(stop_threshold)
    prm.set_track_length(track_length)
    prm.set_cue_conditioned_goal(cue_conditioned_goal)

    prm.set_sorter_name('/' + sorter_name)
    prm.set_output_path(recording_to_process + prm.get_sorter_name())

    raw_position_data, processed_position_data = process_position_data(recording_to_process, prm) #process spatial data for session
    spike_data, bad_clusters = process_firing_properties(recording_to_process, session_type, prm) #process firing properties for all clusters in session

    if len(spike_data) == 0:  # this means that there are no good clusters and the analysis will not run
        save_noisy_cluster_frames(prm, spike_data, raw_position_data,processed_position_data, bad_clusters)
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')
        print('No curated clusters found. Saving dataframe for noisy clusters')
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')
        return
    gc.collect()

    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print(str(len(spike_data)), ' curated clusters found. Processing spatial firing...')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    spike_data = PostSorting.load_snippet_data.get_snippets(spike_data, prm) #load waveform data
    spike_data_movement, spike_data_stationary, spike_data = PostSorting.vr_spatial_firing.process_spatial_firing(spike_data, raw_position_data)
    spike_data = PostSorting.vr_firing_rate_maps.make_firing_field_maps_all(spike_data, raw_position_data, processed_position_data, prm)
    spike_data = PostSorting.vr_FiringMaps_InTime.control_convolution_in_time(spike_data, raw_position_data)

    save_data_frames(prm, spike_data_movement, spike_data_stationary, spike_data, raw_position_data, processed_position_data, bad_clusters)
    make_plots(spike_data, raw_position_data, processed_position_data)
    gc.collect()


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()
    params.stop_threshold = 7.0
    params.cue_conditioned_goal = True
    params.track_length = 300

    recording_folder = '/home/nolanlab/to_sort/recordings/M2_D17_2019-09-25_12-39-02'
    print('Processing ' + str(recording_folder))

    post_process_recording(recording_folder, 'vr')


if __name__ == '__main__':
    main()
