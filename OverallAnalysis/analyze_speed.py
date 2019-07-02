import glob
import pandas as pd
import PostSorting.speed
import OverallAnalysis.folder_path_settings
import OverallAnalysis.analyze_field_correlations
import os

import rpy2.robjects as ro
from rpy2.robjects.packages import importr


local_path_mouse = OverallAnalysis.folder_path_settings.get_local_path() + '/speed/all_mice_df.pkl'
local_path_rat = OverallAnalysis.folder_path_settings.get_local_path() + '/speed/all_rats_df.pkl'
local_path_simulated = OverallAnalysis.folder_path_settings.get_local_path() + '/speed/all_simulated_df.pkl'
path_to_data = OverallAnalysis.folder_path_settings.get_local_path() + '/speed/'
save_output_path = OverallAnalysis.folder_path_settings.get_local_path() + '/speed/'

server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()
server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated()


def add_speed_score_to_spatial_firing(output_path, server_path, animal, spike_sorter='', df_path='/DataFrames'):
    if os.path.exists(output_path):
        spatial_firing = pd.read_pickle(output_path)
        return spatial_firing
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + df_path + '/spatial_firing.pkl'
        position_data_path = recording_folder + spike_sorter + df_path + '/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            position_data = pd.read_pickle(position_data_path)
            if 'grid_score' in spatial_firing:
                if animal == 'rat':
                    spatial_firing = spatial_firing[['session_id', 'cell_id', 'cluster_id', 'firing_times',
                                                    'number_of_spikes', 'hd', 'speed', 'mean_firing_rate']].copy()
                if animal == 'mouse':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'firing_times',
                                                     'number_of_spikes', 'hd', 'speed', 'mean_firing_rate']].copy()
                if animal == 'simulated':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'firing_times',
                                                    'hd', 'hd_spike_histogram', 'speed', 'max_firing_rate_hd']].copy()

                spatial_firing_data = PostSorting.speed.calculate_speed_score(position_data, spatial_firing)
                spatial_firing_data = spatial_firing_data.append(spatial_firing)
    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def process_data():
    add_speed_score_to_spatial_firing(local_path_mouse, server_path_mouse, 'mouse', spike_sorter='/MountainSort', df_path='/DataFrames')


def main():
    process_data()


if __name__ == '__main__':
    main()