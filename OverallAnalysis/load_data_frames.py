import os
import glob
import pandas as pd

server_test_file = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/test_analysis/M5_2018-03-05_13-30-30_of/parameters.txt'
server_path = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/Open_field_opto_tagging_p038/'
local_output_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df_2.pkl'


test_image_path = '/Users/s1466507/Desktop/mouse.jpg'


def load_data_frame_spatial_firing():
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
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'number_of_spikes', 'mean_firing_rate', 'isolation', 'noise_overlap', 'peak_snr', 'hd_correlation_first_vs_second_half', 'hd_correlation_first_vs_second_half_p', 'hd_hist_first_half', 'firing_fields_hd_session', 'hd_hist_second_half', 'watson_test_hd', 'hd_score']].copy()

                # print(spatial_firing.head())
                spatial_firing_data = spatial_firing_data.append(spatial_firing)

                print(spatial_firing_data.head())
    spatial_firing_data.to_pickle(local_output_path)


# for shuffle analysis
def load_data_frame_field_data_frame():
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
                field_data = field_data[['session_id', 'cluster_id', 'field_id', 'indices_rate_map',
                                         'spike_times', 'number_of_spikes_in_field', 'position_x_spikes',
                                         'position_y_spikes', 'hd_in_field_spikes', 'hd_hist_spikes',
                                         'times_session', 'time_spent_in_field', 'position_x_session',
                                         'position_y_session', 'hd_in_field_session', 'hd_hist_session',
                                         'shuffled_data', 'shuffled_means', 'shuffled_std',
                                         'hd_histogram_real_data', 'time_spent_in_bins', 'field_histograms_hz',
                                         'real_and_shuffled_data_differ_bin', 'number_of_different_bins']].copy()

                spatial_firing_data = field_data_combined.append(field_data)

                print(spatial_firing_data.head())
    field_data_combined.to_pickle(local_output_path)


def main():
    if os.path.exists(test_image_path):
        print('I found the test file.')

    if os.path.exists(server_test_file):
        print('I see the server.')

    # load_data_frame_spatial_firing()   # for two-sample watson analysis
    load_data_frame_field_data_frame()


if __name__ == '__main__':
    main()
