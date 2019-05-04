import glob
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import OverallAnalysis.false_positives
import OverallAnalysis.folder_path_settings
import OverallAnalysis.analyze_field_correlations
import os



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

                spatial_firing_data = spatial_firing_data.append(spatial_firing)

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def process_data(animal):
    pass


# load df
# generate shuffled data by shifting spikes in time (20s - recording length - 20s) * 200
# calculate grid score for each shuffle
# find 95th percentile of shuffled grid scores
# check if read grid score is >= to decide if it's a grid cell



def main():
    process_data('mouse')
    process_data('rat')


if __name__ == '__main__':
    main()
