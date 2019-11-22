import glob
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import OverallAnalysis.false_positives
import pandas as pd
import OverallAnalysis.shuffle_cell_analysis
import PostSorting.compare_first_and_second_half
import PostSorting.open_field_firing_maps
import PostSorting.parameters
from scipy import signal

prm = PostSorting.parameters.Parameters()

local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/compare_first_and_second_shuffled/'
local_path_mouse = local_path + 'all_mice_df.pkl'
local_path_mouse_down_sampled = local_path + 'all_mice_df_down_sampled.pkl'
local_path_rat = local_path + 'all_rats_df.pkl'

server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()


def add_cell_types_to_data_frame(spatial_firing):
    cell_type = []
    for index, cell in spatial_firing.iterrows():
        if cell.hd_score >= 0.5 and cell.grid_score >= 0.4:
            cell_type.append('conjunctive')
        elif cell.hd_score >= 0.5:
            cell_type.append('hd')
        elif cell.grid_score >= 0.4:
            cell_type.append('grid')
        else:
            cell_type.append('na')

    spatial_firing['cell type'] = cell_type

    return spatial_firing


def add_combined_id_to_df(spatial_firing):
    animal_ids = [session_id.split('_')[0] for session_id in spatial_firing.session_id.values]
    dates = [session_id.split('_')[1] for session_id in spatial_firing.session_id.values]
    if 'tetrode' in spatial_firing:
        tetrode = spatial_firing.tetrode.values
        cluster = spatial_firing.cluster_id.values

        combined_ids = []
        for cell in range(len(spatial_firing)):
            id = animal_ids[cell] + '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
            combined_ids.append(id)
        spatial_firing['false_positive_id'] = combined_ids
    else:
        cluster = spatial_firing.cluster_id.values
        combined_ids = []
        for cell in range(len(spatial_firing)):
            id = animal_ids[cell] + '-' + dates[cell] + '-Cluster-' + str(cluster[cell])
            combined_ids.append(id)
        spatial_firing['false_positive_id'] = combined_ids

    return spatial_firing


def tag_false_positives(spatial_firing):
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(local_path + 'false_positives_all.txt')
    spatial_firing = add_combined_id_to_df(spatial_firing)
    spatial_firing['false_positive'] = spatial_firing['false_positive_id'].isin(list_of_false_positives)
    return spatial_firing


def load_data(path):
    first_half_spatial_firing = None
    second_half_spatial_firing = None
    first_position = None
    second_position = None
    if os.path.exists(path + '/first_half/DataFrames/spatial_firing.pkl'):
        first_half_spatial_firing = pd.read_pickle(path + '/first_half/DataFrames/spatial_firing.pkl')
    else:
        return None, None, None, None
    if os.path.exists(path + '/second_half/DataFrames/spatial_firing.pkl'):
        second_half_spatial_firing = pd.read_pickle(path + '/second_half/DataFrames/spatial_firing.pkl')
    else:
        return None, None, None, None

    if os.path.exists(path + '/first_half/DataFrames/position.pkl'):
        first_position = pd.read_pickle(path + '/first_half/DataFrames/position.pkl')
    else:
        return None, None, None, None
    if os.path.exists(path + '/second_half/DataFrames/position.pkl'):
        second_position = pd.read_pickle(path + '/second_half/DataFrames/position.pkl')
    else:
        return None, None, None, None
    return first_half_spatial_firing, second_half_spatial_firing, first_position, second_position


def split_in_two(cell):
    cell['position_x_pixels'] = [np.array(cell.position_x.iloc[0]) * prm.get_pixel_ratio() / 100]
    cell['position_y_pixels'] = [np.array(cell.position_y.iloc[0]) * prm.get_pixel_ratio() / 100]
    spike_data_in = cell
    synced_spatial_data_in = pd.DataFrame()
    synced_spatial_data_in['position_x'] = cell.trajectory_x.iloc[0]
    synced_spatial_data_in['position_y'] = cell.trajectory_y.iloc[0]
    synced_spatial_data_in['synced_time'] = cell.trajectory_times.iloc[0]
    synced_spatial_data_in['hd'] = cell.trajectory_hd.iloc[0]
    spike_data_cluster_first, synced_spatial_data_first_half = PostSorting.compare_first_and_second_half.get_half_of_the_data_cell(prm, spike_data_in, synced_spatial_data_in, half='first_half')
    spike_data_cluster_second, synced_spatial_data_second_half = PostSorting.compare_first_and_second_half.get_half_of_the_data_cell(prm, spike_data_in, synced_spatial_data_in, half='second_half')

    synced_spatial_data_first_half['position_x_pixels'] = np.array(synced_spatial_data_first_half.position_x) * prm.get_pixel_ratio() / 100
    synced_spatial_data_first_half['position_y_pixels'] = np.array(synced_spatial_data_first_half.position_y) * prm.get_pixel_ratio() / 100
    synced_spatial_data_second_half['position_x_pixels'] = np.array(synced_spatial_data_second_half.position_x) * prm.get_pixel_ratio() / 100
    synced_spatial_data_second_half['position_y_pixels'] = np.array(synced_spatial_data_second_half.position_y) * prm.get_pixel_ratio() / 100

    first = pd.DataFrame()
    first['session_id'] = cell.session_id.iloc[0]
    first['cluster_id'] = cell.cluster_id.iloc[0]
    first['number_of_spikes'] = [len(spike_data_cluster_first.firing_times)]
    first['firing_times'] = [spike_data_cluster_first.firing_times]
    first['position_x'] = [spike_data_cluster_first.position_x]
    first['position_y'] = [spike_data_cluster_first.position_y]
    first['position_x_pixels'] = [spike_data_cluster_first.position_x_pixels]
    first['position_y_pixels'] = [spike_data_cluster_first.position_y_pixels]
    first['hd'] = [spike_data_cluster_first.hd]

    first['trajectory_x'] = [synced_spatial_data_first_half.position_x]
    first['trajectory_y'] = [synced_spatial_data_first_half.position_y]
    first['trajectory_hd'] = [synced_spatial_data_first_half.hd]
    first['trajectory_times'] = [synced_spatial_data_first_half.synced_time]

    second = pd.DataFrame()
    second['session_id'] = cell.session_id.iloc[0]
    second['cluster_id'] = cell.cluster_id.iloc[0]
    second['number_of_spikes'] = [len(spike_data_cluster_second.firing_times)]
    second['firing_times'] = [spike_data_cluster_second.firing_times]
    second['position_x'] = [spike_data_cluster_second.position_x]
    second['position_y'] = [spike_data_cluster_second.position_y]
    second['position_x_pixels'] = [spike_data_cluster_second.position_x_pixels]
    second['position_y_pixels'] = [spike_data_cluster_second.position_y_pixels]
    second['hd'] = [spike_data_cluster_second.hd]

    second['trajectory_x'] = [synced_spatial_data_second_half.position_x.reset_index(drop=True)]
    second['trajectory_y'] = [synced_spatial_data_second_half.position_y.reset_index(drop=True)]
    second['trajectory_hd'] = [synced_spatial_data_second_half.hd.reset_index(drop=True)]
    second['trajectory_times'] = [synced_spatial_data_second_half.synced_time.reset_index(drop=True)]
    return first, second, synced_spatial_data_first_half, synced_spatial_data_second_half


def process_data(server_path, spike_sorter='/MountainSort', df_path='/DataFrames'):
    all_data = pd.read_pickle(local_path + 'all_mice_df.pkl')
    all_data = add_cell_types_to_data_frame(all_data)
    grid_cells = all_data['cell type'] == 'grid'
    grid_data = all_data[grid_cells]

    iterator = 0
    corr_coefs_mean = []
    corr_stds = []
    for iterator in range(len(grid_data)):
        print(iterator)
        print(grid_data.iloc[0].session_id)
        first_half, second_half, position_first, position_second = split_in_two(grid_data.iloc[iterator:iterator + 1])
        # add rate map to dfs
        # shuffle
        position_heat_map_first, first_half = PostSorting.open_field_firing_maps.make_firing_field_maps(position_first, first_half, prm)
        spatial_firing_first = OverallAnalysis.shuffle_cell_analysis.shuffle_data(first_half, 20, number_of_times_to_shuffle=1000, animal='mouse_first_half', shuffle_type='occupancy')

        position_heat_map_second, second_half = PostSorting.open_field_firing_maps.make_firing_field_maps(position_second, second_half, prm)
        spatial_firing_second = OverallAnalysis.shuffle_cell_analysis.shuffle_data(second_half, 20, number_of_times_to_shuffle=1000, animal='mouse_second_half', shuffle_type='occupancy')
        print('shuffled')
        # compare
        first_shuffles = spatial_firing_first.shuffled_data[0]
        second_shuffles = spatial_firing_second.shuffled_data[0]
        # look at correlations between rows of the two arrays above
        corr = np.corrcoef(first_shuffles, second_shuffles)[1000:, :1000]
        corr_mean = corr.mean()
        corr_std = corr.std()

        corr_coefs_mean.append(corr_mean)
        corr_stds.append(corr_std)

    #todo print and plot results (corr coefs and std)
    print('avg corr correlations ')



def main():
    prm.set_pixel_ratio(440)
    prm.set_sampling_rate(30000)
    process_data(server_path_mouse)


if __name__ == '__main__':
    main()