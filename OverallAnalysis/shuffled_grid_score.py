import glob
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import OverallAnalysis.false_positives
import OverallAnalysis.folder_path_settings
import OverallAnalysis.analyze_field_correlations
import os


local_path_mouse = OverallAnalysis.folder_path_settings.get_local_path() + '/shuffled_grid_analysis/all_mice_df.pkl'
local_path_rat = OverallAnalysis.folder_path_settings.get_local_path() + '/shuffled_grid_analysis/all_rats_df.pkl'
path_to_data = OverallAnalysis.folder_path_settings.get_local_path() + '/shuffled_grid_analysis/'
save_output_path = OverallAnalysis.folder_path_settings.get_local_path() + '/shuffled_grid_analysis/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()


def load_spatial_firing(output_path, server_path, animal, spike_sorter=''):
    if os.path.exists(output_path):
        spatial_firing = pd.read_pickle(output_path)
        return spatial_firing
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_data_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            position_data = pd.read_pickle(position_data_path)
            if 'grid_score' in spatial_firing:
                if animal == 'rat':
                    spatial_firing = spatial_firing[['session_id', 'cell_id', 'cluster_id', 'firing_times',
                                                    'number_of_spikes', 'grid_score']].copy()
                if animal == 'mouse':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'firing_times',
                                                     'number_of_spikes', 'hd', 'speed', 'mean_firing_rate',
                                                     'hd_spike_histogram', 'grid_score']].copy()

                spatial_firing['trajectory_x'] = [position_data.position_x] * len(spatial_firing)
                spatial_firing['trajectory_y'] = [position_data.position_y] * len(spatial_firing)
                spatial_firing_data = spatial_firing_data.append(spatial_firing)

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data

# generate shuffled data by shifting spikes in time (20s - recording length - 20s) * 200
def shuffle_data(cell, sampling_rate):
    firing_times = cell.firing_times
    times_to_shift_by = np.random.uniform(20*sampling_rate, firing_times[-1] * sampling_rate, 200) / sampling_rate
    firing_times_reshaped = firing_times.reshape(1, len(firing_times)) / 30000
    shuffled_times = np.repeat(firing_times_reshaped, 200, axis=0).T + times_to_shift_by % firing_times[-1]
    return shuffled_times


def get_position_for_shuffled_times(shuffled_times):
    position_x = []
    position_y = []
    return position_x, position_y


def process_data(animal):
    if animal == 'mouse':
        spatial_firing = load_spatial_firing(local_path_mouse, server_path_mouse, animal, spike_sorter='\MountainSort')
        sampling_rate = 30000
    else:
        spatial_firing = load_spatial_firing(local_path_rat, server_path_rat, animal, spike_sorter='')
        sampling_rate = 50000

    for index, cell in spatial_firing.iterrows():
        if ~np.isnan(cell.grid_score):
            shuffled_times_seconds = shuffle_data(cell, sampling_rate)



# calculate grid score for each shuffle
# find 95th percentile of shuffled grid scores
# check if read grid score is >= to decide if it's a grid cell


def main():
    process_data('mouse')
    process_data('rat')


if __name__ == '__main__':
    main()
