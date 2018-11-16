import os
import glob
import pandas as pd

server_test_file = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/test_analysis/M5_2018-03-05_13-30-30_of/parameters.txt'
server_path = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/Open_field_opto_tagging_p038/'
local_output_path = '/Users/s1466507/Documents/Ephys/test_overall_analysis/test_df.pkl'


test_image_path = '/Users/s1466507/Desktop/mouse.jpg'

if os.path.exists(test_image_path):
    print('I found the test file.')

if os.path.exists(server_test_file):
    print('I see the server.')

spatial_firing_data = pd.DataFrame()
for recording_folder in glob.glob(server_path + '*'):
    os.path.isdir(recording_folder)
    data_frame_path = recording_folder + '/DataFrames/spatial_firing.pkl'
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
        if 'hd_hist_first_half' in spatial_firing:
            spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'number_of_spikes', 'mean_firing_rate', 'isolation', 'noise_overlap', 'peak_snr', 'hd_correlation_first_vs_second_half', 'hd_correlation_first_vs_second_half_p', 'hd_hist_first_half', 'firing_fields_hd_session', 'hd_hist_second_half', 'watson_test_hd']].copy()

            # print(spatial_firing.head())
            spatial_firing_data = spatial_firing_data.append(spatial_firing)

            print(spatial_firing_data.head())
spatial_firing_data.to_pickle(local_output_path)
