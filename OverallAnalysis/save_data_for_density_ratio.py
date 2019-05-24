import glob
import matplotlib.pylab as plt
import math_utility
import math
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import pandas as pd
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots
import plot_utility
from rpy2 import robjects as robj
from scipy.stats import circstd
from rpy2.robjects import pandas2ri
import scipy.stats
import seaborn
import PostSorting.compare_first_and_second_half
import PostSorting.parameters

local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/data_for_modeling/density_ratio/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()


def save_trajectory_field_data_for_cell(position, spatial_firing):
    for index, cluster in spatial_firing.iterrows():
        cluster_id = spatial_firing.cluster_id[index]
        session_id = spatial_firing.session_id[index]
        number_of_firing_fields = len(spatial_firing.firing_fields[index])
        if number_of_firing_fields > 0:
            hd = position.hd
            times_position = position.synced_time
            field_id = np.zeros(len(hd))
            for field in range(len(cluster.firing_fields)):
                occupancy_times = cluster.times_in_session_fields[field]
                mask_for_occupancy = np.in1d(times_position, occupancy_times)
                field_id[mask_for_occupancy] = field + 1

            trajectory_df_to_save = pd.DataFrame()
            trajectory_df_to_save['hd'] = hd
            trajectory_df_to_save['field_id'] = field_id
            trajectory_df_to_save['cluster_id'] = cluster.cluster_id
            trajectory_df_to_save['session_id'] = cluster.session_id
            trajectory_df_to_save.to_csv(local_path + session_id + str(cluster_id) + '.csv')


def save_trajectory_field_data_for_cell_spikes(spatial_firing):
    for index, cluster in spatial_firing.iterrows():
        cluster_id = spatial_firing.cluster_id[index]
        session_id = spatial_firing.session_id[index]
        number_of_firing_fields = len(spatial_firing.firing_fields[index])
        if number_of_firing_fields > 0:
            hd = spatial_firing.hd[cluster_id - 1]
            all_spike_times = spatial_firing.firing_times[cluster_id - 1]
            field_id = np.zeros(len(hd))
            for field in range(len(cluster.firing_fields)):
                spike_times_field = cluster.spike_times_in_fields[field]
                mask_for_occupancy = np.in1d(all_spike_times, spike_times_field)
                field_id[mask_for_occupancy] = field + 1

            spike_df_to_save = pd.DataFrame()
            spike_df_to_save['hd'] = hd
            spike_df_to_save['field_id'] = field_id
            spike_df_to_save['cluster_id'] = cluster.cluster_id
            spike_df_to_save['session_id'] = cluster.session_id
            spike_df_to_save.to_csv(local_path + session_id + str(cluster_id) + '_spikes.csv')


def load_data_frame_spatial_firing_modeling(output_path, server_path, spike_sorter='/MountainSort'):
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        firing_data_frame_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        if os.path.exists(firing_data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(firing_data_frame_path)
            position = pd.read_pickle(position_path)
            save_trajectory_field_data_for_cell_spikes(spatial_firing)
            save_trajectory_field_data_for_cell(position, spatial_firing)



def process_data():
    load_data_frame_spatial_firing_modeling(local_path, server_path_mouse, spike_sorter='/MountainSort')



def main():
    process_data()


if __name__ == '__main__':
    main()