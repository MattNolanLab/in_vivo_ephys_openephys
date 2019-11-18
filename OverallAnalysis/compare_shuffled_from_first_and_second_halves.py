import glob
import os
import OverallAnalysis.folder_path_settings
import OverallAnalysis.false_positives
import pandas as pd
import OverallAnalysis.shuffle_cell_analysis
import PostSorting.compare_first_and_second_half
import PostSorting.parameters

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


def process_data(server_path, spike_sorter='/MountainSort', df_path='/DataFrames'):
    for recording_folder in glob.glob(server_path + '*'):
        print(recording_folder)
        firing_data_frame_path = recording_folder + spike_sorter + df_path + '/spatial_firing.pkl'
        if os.path.exists(firing_data_frame_path):
            print('I found a firing data frame.')
            try:
                spatial_firing = pd.read_pickle(firing_data_frame_path)
            except:
                print('could not read pickle')
            if 'grid_score' in spatial_firing:
                spatial_firing = add_cell_types_to_data_frame(spatial_firing)
                if 'grid' not in list(spatial_firing['cell type']):
                    print('no grid cells here')
                    continue
                # os.path.isdir(recording_folder)
                first_half_spatial_firing, second_half_spatial_firing, first_position, second_position = load_data(recording_folder)
                if first_half_spatial_firing is None:
                    continue
                first_half = tag_false_positives(first_half_spatial_firing)
                second_half = tag_false_positives(second_half_spatial_firing)
                first_half['cell type'] = spatial_firing['cell type']
                first_half['trajectory_hd'] = [first_position.hd] * len(first_half)
                first_half['trajectory_x'] = [first_position.position_x] * len(first_half)
                first_half['trajectory_y'] = [first_position.position_y] * len(first_half)
                first_half['trajectory_times'] = [first_position.synced_time] * len(first_half)

                second_half['cell type'] = spatial_firing['cell type']
                second_half['trajectory_hd'] = [second_position.hd] * len(second_half)
                second_half['trajectory_x'] = [second_position.position_x] * len(second_half)
                second_half['trajectory_y'] = [second_position.position_y] * len(second_half)
                second_half['trajectory_times'] = [second_position.synced_time] * len(second_half)

                spatial_firing = OverallAnalysis.shuffle_cell_analysis.shuffle_data(first_half, 20, number_of_times_to_shuffle=1000, animal='mouse', shuffle_type='occupancy')

                for index, cluster in first_half_spatial_firing.iterrows():

                    pass
                        # generate shuffled data for both halves
                        # bin / make hd histograms

                        # get 1000 correlation values between first and second halves
                        # save correlation values in df + save mean and sd locally


def main():
    prm.set_pixel_ratio(440)
    process_data(server_path_mouse)


if __name__ == '__main__':
    main()