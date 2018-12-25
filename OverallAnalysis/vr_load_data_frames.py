import os
import glob
import pandas as pd

recording_folder = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D27_2018-10-05_11-17-55' # test recording
local_output_path = '/Users/sarahtennant/Work/Analysis/Opto_data/PVCre1/M1_D27_2018-10-05_11-17-55/Dataframes/spatial_firing.pkl'
data_frame_path = recording_folder + '/DataFrames/spatial_firing.pkl'


if os.path.exists(recording_folder):
    print('I found the test file.')

if os.path.exists(local_output_path):
    print('I found the output folder.')

spatial_firing_data = pd.DataFrame()

os.path.isdir(recording_folder)
if os.path.exists(data_frame_path):
    print('I found a firing data frame.')
    spatial_firing = pd.read_pickle(data_frame_path)
    '''
    'session_id' 'cluster_id' 'tetrode' 'primary_channel' 'firing_times'
     'firing_times_opto' 'number_of_spikes' 'mean_firing_rate' 'isolation'
     'noise_overlap' 'peak_snr' 'peak_amp' 'random_snippets' 'x_position_cm'
     'trial_number' 'trial_type' 'normalised_b_spike_number' 'normalised_nb_spike_number' 'normalised_p_spike_number'


    '''

spatial_firing_data.to_pickle(local_output_path)
