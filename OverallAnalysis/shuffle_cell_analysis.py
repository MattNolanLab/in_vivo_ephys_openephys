import glob
import numpy as np
import os
import pandas as pd
import shutil


local_path_mouse = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/shuffled_analysis_cell/all_mice_df.pkl'
local_path_rat = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/shuffled_analysis_cell/all_rats_df.pkl'
analysis_path = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/shuffled_analysis_cell/'

server_path_mouse = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/Open_field_opto_tagging_p038/'
server_path_rat = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/grid_field_analysis/moser_data/Sargolini/all_data/'


def load_data_frame_spatial_firing(output_path, server_path, spike_sorter='/MountainSort'):
    if os.path.exists(output_path):
        spatial_firing = pd.read_pickle(output_path)
        return spatial_firing
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        firing_data_frame_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        if os.path.exists(firing_data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(firing_data_frame_path)
            position = pd.read_pickle(position_path)

            if 'position_x' in spatial_firing:
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'number_of_spikes', 'mean_firing_rate', 'hd_score', 'position_x', 'position_y', 'hd']].copy()
                spatial_firing['trajectory_hd'] = [position.hd] * len(spatial_firing)
                spatial_firing_data = spatial_firing_data.append(spatial_firing)
                print(spatial_firing_data.head())

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def get_random_indices_for_shuffle(cell, number_of_times_to_shuffle):
    number_of_spikes_in_field = cell['number_of_spikes']
    length_of_recording = len(cell.trajectory_hd)
    shuffle_indices = np.random.randint(0, length_of_recording, size=(number_of_times_to_shuffle, number_of_spikes_in_field))
    return shuffle_indices


# todo rewrite this to work for cells
# add shuffled data to data frame as a new column for each field
def shuffle_data(spatial_firing, number_of_bins, number_of_times_to_shuffle=1000):
    if os.path.exists(analysis_path + 'shuffle_analysis') is True:
        shutil.rmtree(analysis_path + 'shuffle_analysis')
    os.makedirs(analysis_path + 'shuffle_analysis')

    shuffled_histograms_all = []
    for index, cell in spatial_firing.iterrows():
        print('I will shuffle data.')
        shuffled_histograms = np.zeros((number_of_times_to_shuffle, number_of_bins))
        shuffle_indices = get_random_indices_for_shuffle(cell, number_of_times_to_shuffle)
        for shuffle in range(number_of_times_to_shuffle):
            shuffled_hd = cell['trajectory_hd'][shuffle_indices[shuffle]]
            hist, bin_edges = np.histogram(shuffled_hd, bins=number_of_bins, range=(0, 6.28))  # from 0 to 2pi
            shuffled_histograms[shuffle, :] = hist
        shuffled_histograms_all.append(shuffled_histograms)
    spatial_firing['shuffled_data'] = shuffled_histograms_all
    return spatial_firing


def process_data(spatial_firing):
    spatial_firing = shuffle_data(spatial_firing, 20, number_of_times_to_shuffle=1000)


def main():
    spatial_firing_all_mice = load_data_frame_spatial_firing(local_path_mouse, server_path_mouse, spike_sorter='/MountainSort')
    spatial_firing_all_rats = load_data_frame_spatial_firing(local_path_rat, server_path_rat, spike_sorter='')
    process_data(spatial_firing_all_mice)
    process_data(spatial_firing_all_rats)


if __name__ == '__main__':
    main()
