import pandas as pd
import os
import PostSorting.open_field_spatial_firing
import PostSorting.speed
import PostSorting.open_field_head_direction
import PostSorting.open_field_firing_maps
import PostSorting.open_field_grid_cells
import PostSorting.open_field_border_cells
import numpy as np
import settings
import time
from PostSorting import parameters

prm = parameters.Parameters()

def run_parallel_of_shuffle(single_shuffle, synced_spatial_data, prm):
    prm.set_sampling_rate(30000)
    prm.set_pixel_ratio(440)

    single_shuffle = PostSorting.open_field_spatial_firing.process_spatial_firing(single_shuffle, synced_spatial_data)
    single_shuffle = PostSorting.speed.calculate_speed_score(synced_spatial_data, single_shuffle, settings.gauss_sd_for_speed_score, settings.sampling_rate)
    _, single_shuffle = PostSorting.open_field_head_direction.process_hd_data(single_shuffle, synced_spatial_data, prm)
    position_heatmap, single_shuffle = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, single_shuffle, prm)
    single_shuffle = PostSorting.open_field_grid_cells.process_grid_data(single_shuffle)
    single_shuffle = PostSorting.open_field_firing_maps.calculate_spatial_information(single_shuffle, position_heatmap)
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

def one_job_shuffle_parallel(recording_path, n_shuffles):
    '''
    creates a single shuffle of each cell and saves it in recording/sorter/dataframes/shuffles/
    :param recording_path: path to a recording folder
    :param shuffle_id: integer id of a single shuffle
    '''
    time0 = time.time()

    spike_data_spatial = pd.read_pickle(recording_path+"/MountainSort/DataFrames/spatial_firing.pkl")
    synced_spatial_data = pd.read_pickle(recording_path+"/MountainSort/DataFrames/position.pkl")

    if os.path.isfile(recording_path+"/MountainSort/DataFrames/shuffles/shuffle.pkl"):
        shuffle = pd.read_pickle(recording_path+"/MountainSort/DataFrames/shuffles/shuffle.pkl")
    else:
        shuffle = pd.DataFrame()

    for cluster_index, cluster_id in enumerate(spike_data_spatial.cluster_id):
        cluster_spike_data = spike_data_spatial[(spike_data_spatial["cluster_id"] == cluster_id)]

        if os.path.isfile(recording_path+"/MountainSort/DataFrames/shuffles/shuffle.pkl"):
            n_shuffles_pre_computed = len(shuffle[shuffle["cluster_id"] == cluster_id])
        else:
            n_shuffles_pre_computed = 0

        shuffles_to_run = n_shuffles-n_shuffles_pre_computed

        if shuffles_to_run > 1:
            for i in range(shuffles_to_run):
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
    n_shuffles = os.environ['SHUFFLE_NUMBER']
    one_job_shuffle_parallel(recording_path, n_shuffles)
    #=========================================================================================#
    #=========================================================================================#

if __name__ == '__main__':
    main()