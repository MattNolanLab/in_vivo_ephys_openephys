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
import PostSorting.vr_speed_analysis
import gc
import PostSorting.vr_cued
import PostSorting.theta_modulation
import PostSorting.vr_grid_cells
import PostSorting.lfp

prm = PostSorting.parameters.Parameters()


def initialize_parameters(recording_to_process):
    prm.set_is_ubuntu(True)
    prm.set_sampling_rate(30000)
    prm.set_downsampled_rate(1000)
    prm.set_local_recording_folder_path(recording_to_process)
    prm.set_opto_channel('100_ADC3.continuous')
    prm.set_stop_threshold(0.7)  # speed is given in cm/200ms 0.7*1/2000
    prm.set_movement_channel('100_ADC2.continuous')
    prm.set_first_trial_channel('100_ADC4.continuous')
    prm.set_second_trial_channel('100_ADC5.continuous')
    prm.set_goal_location_chennl('100_ADC7.continuous')
    prm.set_ephys_channels(PostSorting.load_firing_data.available_ephys_channels(recording_to_process, prm))
    prm.set_file_path(recording_to_process)
    prm.set_local_recording_folder_path(recording_to_process)
    prm.set_ms_tmp_path('/tmp/mountainlab/')


def process_position_data(recording_to_process, prm):
    raw_position_data, position_data = PostSorting.vr_sync_spatial_data.syncronise_position_data(recording_to_process, prm)
    processed_position_data = PostSorting.vr_spatial_data.process_position(raw_position_data, prm, recording_to_process)
    return raw_position_data, processed_position_data, position_data


def process_firing_properties(recording_to_process, session_type, prm):
    spike_data = PostSorting.load_firing_data.create_firing_data_frame(recording_to_process, session_type, prm)
    spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, prm)
    spike_data = PostSorting.temporal_firing.correct_for_stitch(spike_data, prm)
    spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, prm)
    return spike_data, bad_clusters


def save_data_frames(prm, spatial_firing_movement=None, spatial_firing_stationary=None, spatial_firing=None,
                     raw_position_data=None, processed_position_data=None, position_data=None, snippet_data=None, bad_clusters=None,
                     lfp_data=None):
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    if spatial_firing_movement is not None:
        spatial_firing_movement.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing_movement.pkl')
    if spatial_firing_stationary is not None:
        spatial_firing_stationary.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing_stationary.pkl')
    if spatial_firing is not None:
        spatial_firing.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing.pkl')
    if raw_position_data is not None:
        print(" I am not saving the raw positional pickle at the moment")
        #raw_position_data.to_pickle(prm.get_output_path() + '/DataFrames/raw_position_data.pkl')
    if processed_position_data is not None:
        processed_position_data.to_pickle(prm.get_output_path() + '/DataFrames/processed_position_data.pkl')
    if position_data is not None:
        position_data.to_pickle(prm.get_output_path() + '/DataFrames/position_data.pkl')
    if bad_clusters is not None:
        bad_clusters.to_pickle(prm.get_output_path() + '/DataFrames/noisy_clusters.pkl')
    if snippet_data is not None:
        snippet_data.to_pickle(prm.get_output_path() + '/DataFrames/snippet_data.pkl')
    if lfp_data is not None:
        lfp_data.to_pickle(prm.get_output_path() + "/DataFrames/lfp_data.pkl")

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
            print('Unexpected / incorrect tag in the third line of parameters file')
            unexpected_tag = True
    return stop_threshold, track_length, cue_conditioned_goal

def post_process_recording(recording_to_process, session_type, running_parameter_tags=False,
                           sorter_name='MountainSort', stitchpoint=None, paired_order=None, total_length=None):

    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    stop_threshold, track_length, cue_conditioned_goal = process_running_parameter_tag(running_parameter_tags)
    prm.set_paired_order(paired_order)
    prm.set_stitch_point(stitchpoint)
    prm.set_stop_threshold(stop_threshold)
    prm.set_track_length(track_length)
    prm.set_vr_grid_analysis_bin_size(5)
    prm.set_cue_conditioned_goal(cue_conditioned_goal)
    if total_length is not None:
        prm.set_total_length_sampling_points(total_length/prm.get_sampling_rate())

    prm.set_sorter_name('/' + sorter_name)
    prm.set_output_path(recording_to_process + prm.get_sorter_name())

    lfp_data = PostSorting.lfp.process_lfp(recording_to_process, session_type=session_type, prm=prm)
    raw_position_data, processed_position_data, position_data = process_position_data(recording_to_process, prm)
    spike_data, bad_clusters = process_firing_properties(recording_to_process, session_type, prm)
    snippet_data = PostSorting.load_snippet_data.get_snippets(spike_data, prm, random_snippets=False)

    if len(spike_data) == 0:  # this means that there are no good clusters and the analysis will not run
        PostSorting.vr_make_plots.make_plots(raw_position_data, processed_position_data, spike_data=None, prm=prm)

        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')
        print('No curated clusters found. Saving dataframe for noisy clusters...')
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')

    else:
        PostSorting.vr_make_plots.make_plots(raw_position_data, processed_position_data, spike_data=spike_data, prm=prm)

        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')
        print(str(len(spike_data)), ' curated clusters found. Processing spatial firing...')
        print('-------------------------------------------------------------')
        print('-------------------------------------------------------------')

        spike_data = PostSorting.load_snippet_data.get_snippets(spike_data, prm, random_snippets=True)
        spike_data_movement, spike_data_stationary, spike_data = PostSorting.vr_spatial_firing.process_spatial_firing(spike_data, raw_position_data, prm)
        spike_data = PostSorting.vr_grid_cells.process_vr_grid(spike_data, position_data, prm.get_vr_grid_analysis_bin_size(), prm)
        spike_data = PostSorting.vr_firing_rate_maps.make_firing_field_maps_all(spike_data, raw_position_data, processed_position_data, prm)
        spike_data = PostSorting.vr_FiringMaps_InTime.control_convolution_in_time(spike_data, raw_position_data)
        spike_data = PostSorting.theta_modulation.calculate_theta_index(spike_data, prm)

    save_data_frames(prm,
                     spatial_firing=spike_data,
                     raw_position_data=raw_position_data,
                     processed_position_data=processed_position_data,
                     position_data=position_data,
                     snippet_data=snippet_data,
                     bad_clusters=bad_clusters,
                     lfp_data=lfp_data)
    gc.collect()


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()
    params.stop_threshold = 4.7
    params.cue_conditioned_goal = False
    params.track_length = 200

    recording_folder = "/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D9_2020-11-08_14-37-47"

    print('Processing ' + str(recording_folder))

    post_process_recording(recording_folder, 'vr')


if __name__ == '__main__':
    main()
