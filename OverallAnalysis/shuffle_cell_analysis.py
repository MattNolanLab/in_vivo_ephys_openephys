import glob
import os
import pandas as pd


local_path_mouse = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/shuffled_analysis_cell/all_mice_df.pkl'
local_path_rat = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/shuffled_analysis_cell/all_rats_df.pkl'

server_path_mouse = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/Open_field_opto_tagging_p038/'
server_path_rat = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/grid_field_analysis/moser_data/Sargolini/all_data/'


def load_data_frame_spatial_firing(output_path, server_path, spike_sorter='/MountainSort'):
    if os.path.exists(output_path):
        spatial_firing = pd.read_pickle(output_path)
        return spatial_firing
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)

            if 'position_x' in spatial_firing:
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'number_of_spikes', 'mean_firing_rate', 'hd_score', 'position_x', 'position_y', 'hd']].copy()
                spatial_firing_data = spatial_firing_data.append(spatial_firing)

                print(spatial_firing_data.head())

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def process_data(spatial_firing):
    pass


def main():
    spatial_firing_all_mice = load_data_frame_spatial_firing(local_path_mouse, server_path_mouse, spike_sorter='/MountainSort')
    spatial_firing_all_rats = load_data_frame_spatial_firing(local_path_rat, server_path_rat, spike_sorter='')
    process_data(spatial_firing_all_mice)
    process_data(spatial_firing_all_rats)


if __name__ == '__main__':
    main()
