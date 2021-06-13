import pandas as pd
from joblib import Parallel, delayed
import os
import multiprocessing
from control_sorting_analysis import get_session_type
import PostSorting
import numpy as np
import settings
import sys
import traceback
import time
import gc

prm = PostSorting.parameters.Parameters()

def run_parallel_of_shuffle(single_shuffle, synced_spatial_data, prm):
    prm.set_sampling_rate(30000)
    prm.set_pixel_ratio(440)

    single_shuffle = PostSorting.open_field_spatial_firing.process_spatial_firing(single_shuffle, synced_spatial_data)
    single_shuffle = PostSorting.speed.calculate_speed_score(synced_spatial_data, single_shuffle, settings.gauss_sd_for_speed_score, settings.sampling_rate)
    _, single_shuffle = PostSorting.open_field_head_direction.process_hd_data(single_shuffle, synced_spatial_data, prm)
    _, single_shuffle = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, single_shuffle, prm)
    single_shuffle = PostSorting.open_field_firing_maps.calculate_spatial_information(single_shuffle)
    single_shuffle = PostSorting.open_field_grid_cells.process_grid_data(single_shuffle)
    single_shuffle = PostSorting.open_field_border_cells.process_border_data(single_shuffle)

    single_shuffle = single_shuffle[["cluster_id", "shuffle_id", "mean_firing_rate", "speed_score", "speed_score_p_values", "hd_score", "rayleigh_score", "spatial_information_score", "grid_score", "border_score"]]
    return single_shuffle

def generate_shuffled_times(cluster_firing, n_shuffles):
    cluster_firing = cluster_firing[["cluster_id", "firing_times", "mean_firing_rate", "recording_length_sampling_points"]]

    shuffle_firing = pd.DataFrame()
    for i in range(n_shuffles):
        shuffle = cluster_firing.copy()
        firing_times = shuffle["firing_times"].to_numpy()[0]
        random_firing_additions = np.random.randint(low=int(20*settings.sampling_rate),
                                                    high=int(580*settings.sampling_rate), size=len(firing_times))

        shuffled_firing_times = firing_times + random_firing_additions
        recording_length = int(cluster_firing["recording_length_sampling_points"].iloc[0])
        shuffled_firing_times[shuffled_firing_times > recording_length] = shuffled_firing_times[shuffled_firing_times > recording_length] - recording_length # wrap around the firing times that exceed the length of the recording
        shuffle["firing_times"] = [shuffled_firing_times]

        shuffle_firing = pd.concat([shuffle_firing, shuffle], ignore_index=True)

    shuffle_firing["shuffle_id"] = np.arange(0, n_shuffles)
    return shuffle_firing

def one_job_shuffle_parallel(recording_path):
    '''
    creates a single shuffle of each cell and saves it in recording/sorter/dataframes/shuffles/
    :param recording_path: path to a recording folder
    :param shuffle_id: integer id of a single shuffle
    '''
    time0 = time.time()
    N_SHUFFLES = 1000

    spike_data_spatial = pd.read_pickle(recording_path+"/MountainSort/DataFrames/spatial_firing.pkl")
    synced_spatial_data = pd.read_pickle(recording_path+"/MountainSort/DataFrames/position.pkl")

    shuffle = pd.DataFrame()
    for cluster_index, cluster_id in enumerate(spike_data_spatial.cluster_id):
        cluster_spike_data = spike_data_spatial[(spike_data_spatial["cluster_id"] == cluster_id)]

        for i in range(N_SHUFFLES):
            shuffled_cluster_spike_data = generate_shuffled_times(cluster_spike_data, n_shuffles=1)
            shuffled_cluster_spike_data = run_parallel_of_shuffle(shuffled_cluster_spike_data, synced_spatial_data, prm)

            shuffle = pd.concat([shuffle, shuffled_cluster_spike_data], ignore_index=True)
            print(i, " shuffle complete")


            if (time.time()-time0) > 171000: # time in seconds of 47hrs 30 minutes
                finish(shuffle, recording_path)
                return

    finish(shuffle, recording_path)
    return

def finish(shuffle, recording_path):

    if not os.path.exists(recording_path+"/MountainSort/DataFrames/shuffles"):
        os.mkdir(recording_path+"/MountainSort/DataFrames/shuffles")

    shuffle.to_pickle(recording_path+"/MountainSort/DataFrames/shuffles/shuffle.pkl")


def run_shuffle_analysis_vr(recording, n_shuffles, prm):
    return

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #========================FOR RUNNING ON FROM TERMINAL=====================================#
    #=========================================================================================#
    recording_path = os.environ['RECORDING_PATH']
    one_job_shuffle_parallel(recording_path)
    #=========================================================================================#
    #=========================================================================================#

if __name__ == '__main__':
    main()