import os
import glob
import pandas as pd
import OverallAnalysis.false_positives
import PostSorting.open_field_grid_cells
import OverallAnalysis.analyze_hd_from_whole_session

server_test_file = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/test_analysis/M5_2018-03-05_13-30-30_of/parameters.txt'
# server_path = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/Open_field_opto_tagging_p038/'
server_path = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/Open_field_opto_tagging_p038/'
local_output_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df_2.pkl'

false_positives_path = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/data_for_modeling/false_positives_all.txt'


test_image_path = '/Users/s1466507/Desktop/mouse.jpg'


def load_data_frame_spatial_firing(output_path):
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            '''
            'session_id' 'cluster_id' 'tetrode' 'primary_channel' 'firing_times'
             'firing_times_opto' 'number_of_spikes' 'mean_firing_rate' 'isolation'
             'noise_overlap' 'peak_snr' 'peak_amp' 'random_snippets' 'position_x'
             'position_x_pixels' 'position_y' 'position_y_pixels' 'hd' 'speed'
             'hd_spike_histogram' 'max_firing_rate_hd' 'preferred_HD' 'hd_score'
             'firing_maps' 'max_firing_rate' 'firing_fields' 'field_max_firing_rate'
             'firing_fields_hd_session' 'firing_fields_hd_cluster' 'field_hd_max_rate'
             'field_preferred_hd' 'field_hd_score' 'number_of_spikes_in_fields'
             'time_spent_in_fields_sampling_points' 'spike_times_in_fields'
             'times_in_session_fields' 'field_corr_r' 'field_corr_p'
             'hd_correlation_first_vs_second_half'
             'hd_correlation_first_vs_second_half_p' 'hd_hist_first_half'
             'hd_hist_second_half'

            '''
            if ('hd_hist_first_half' in spatial_firing) and ('watson_test_hd' in spatial_firing):
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'number_of_spikes', 'mean_firing_rate', 'isolation', 'noise_overlap', 'peak_snr', 'hd_correlation_first_vs_second_half', 'hd_correlation_first_vs_second_half_p', 'hd_hist_first_half', 'firing_fields_hd_session', 'hd_hist_second_half', 'watson_test_hd', 'hd_score', 'hd', 'kuiper_cluster', 'watson_cluster', 'firing_maps']].copy()

                # print(spatial_firing.head())
                spatial_firing_data = spatial_firing_data.append(spatial_firing)

                print(spatial_firing_data.head())
    spatial_firing_data.to_pickle(output_path)


def load_data_frame(output_path):
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            if 'position_x' in spatial_firing:
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'number_of_spikes', 'mean_firing_rate', 'isolation', 'noise_overlap', 'peak_snr', 'firing_times', 'position_x', 'position_y', 'hd', 'speed', 'firing_maps']].copy()

                # print(spatial_firing.head())
                spatial_firing_data = spatial_firing_data.append(spatial_firing)

            print(spatial_firing_data.head())
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(false_positives_path)
    spatial_firing_data = OverallAnalysis.analyze_hd_from_whole_session.add_combined_id_to_df(spatial_firing_data)
    spatial_firing_data['false_positive'] = spatial_firing_data['false_positive_id'].isin(list_of_false_positives)
    spatial_firing_data = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing_data)
    spatial_firing_data = spatial_firing_data.drop(columns="false_positive_id")
    spatial_firing_data.to_pickle(output_path)


# for shuffle analysis
def load_data_frame_field_data_frame(output_path):
    field_data_combined = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + '/MountainSort/DataFrames/shuffled_fields.pkl'
        if os.path.exists(data_frame_path):
            print('I found a field data frame.')
            field_data = pd.read_pickle(data_frame_path)
            '''
            'session_id', 'cluster_id', 'field_id', 'indices_rate_map',
            'spike_times', 'number_of_spikes_in_field', 'position_x_spikes',
            'position_y_spikes', 'hd_in_field_spikes', 'hd_hist_spikes',
            'times_session', 'time_spent_in_field', 'position_x_session',
            'position_y_session', 'hd_in_field_session', 'hd_hist_session',
            'shuffled_data', 'shuffled_means', 'shuffled_std',
            'hd_histogram_real_data', 'time_spent_in_bins', 'field_histograms_hz',
            'real_and_shuffled_data_differ_bin', 'number_of_different_bins'
            '''
            if 'shuffled_data' in field_data:
                field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'indices_rate_map',
                                         'spike_times', 'number_of_spikes_in_field', 'position_x_spikes',
                                         'position_y_spikes', 'hd_in_field_spikes', 'hd_hist_spikes',
                                         'times_session', 'time_spent_in_field', 'position_x_session',
                                         'position_y_session', 'hd_in_field_session', 'hd_hist_session',
                                         'shuffled_means', 'shuffled_std',
                                         'hd_histogram_real_data', 'time_spent_in_bins', 'field_histograms_hz',
                                         'real_and_shuffled_data_differ_bin', 'number_of_different_bins', 'number_of_different_bins_shuffled', 'number_of_different_bins_bh', 'number_of_different_bins_holm', 'number_of_different_bins_shuffled_corrected_p']].copy()

                field_data_combined = field_data_combined.append(field_data_to_combine)
                print(field_data_combined.head())
    field_data_combined.to_pickle(output_path)


def load_position_data(output_path):
    spatial_data = pd.DataFrame()
    session_id = []
    synced_time = []
    position_x = []
    position_y = []
    hd = []
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + '/MountainSort/DataFrames/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            position = pd.read_pickle(data_frame_path)
            synced_time.append(position.synced_time)
            position_x.append(position.position_x)
            position_y.append(position.position_y)
            hd.append(position.hd)
            session_id.append(data_frame_path.split('/')[-4].split('\\')[-1])
            # spatial_data = spatial_data.append(position)
    spatial_data['session_id'] = session_id
    spatial_data['synced_time'] = synced_time
    spatial_data['position_x'] = position_x
    spatial_data['position_y'] = position_y
    spatial_data['hd'] = hd
    spatial_data.to_pickle(output_path)


def main():
    if os.path.exists(test_image_path):
        print('I found the test file.')

    if os.path.exists(server_test_file):
        print('I see the server.')

    # load_data_frame('/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/data_for_modeling/spatial_firing_all_mice.pkl')
    load_position_data('/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/data_for_modeling/trajectory_all_mice.pkl')
    # load_data_frame_spatial_firing('/Users/s1466507/Documents/Ephys/recordings/all_mice_df_all2.pkl')   # for two-sample watson analysis
    # load_data_frame_field_data_frame('/Users/s1466507/Documents/Ephys/recordings/shuffled_field_data_all_mice.pkl')  # for shuffled field analysis


if __name__ == '__main__':
    main()
