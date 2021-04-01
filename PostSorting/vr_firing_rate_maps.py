import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import PostSorting.vr_sync_spatial_data
import os

def gaussian_kernel(kernx):
    kerny = np.exp(np.power(kernx, 2)/2 * (-1))
    return kerny

def quick_spike_plot(spike_data, prm, trials, locations, cluster_index):
    save_path = prm.get_output_path() + '/Figures/spike_number'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(locations, trials, 'o', markersize=0.5)
    plt.savefig(prm.get_output_path() + '/Figures/spike_number/' + '/' + spike_data.session_id.iloc[cluster_index] + str(spike_data.cluster_id.iloc[cluster_index]) + '.png')
    plt.close()
    return

def get_number_of_bins(prm):
    # bin number is equal to the track length, such that theres one bin per cm
    number_of_bins = prm.track_length
    return number_of_bins

def get_bin_size(prm, numbers_of_bins):
    bin_size = prm.track_length/numbers_of_bins
    return bin_size

def get_total_bin_times(binned_times_collumn):
    # this function adds all the binned times per trial to give the total
    # time spent in a location bin for a given processed_position_data-like dataframe
    total_bin_times = np.zeros(len(binned_times_collumn.iloc[0]))
    for i in range(len(binned_times_collumn)):
        total_bin_times += np.nan_to_num(binned_times_collumn.iloc[i])
    return total_bin_times


def make_firing_field_maps(spike_data, processed_position_data, bin_size_cm, track_length):

    beaconed_processed_position_data = processed_position_data[processed_position_data["trial_type"] == 0]
    non_beaconed_processed_position_data = processed_position_data[processed_position_data["trial_type"] == 1]
    probe_processed_position_data = processed_position_data[processed_position_data["trial_type"] == 2]

    bins = np.arange(0, track_length, bin_size_cm)

    beaconed_firing_rate_map = []
    non_beaconed_firing_rate_map = []
    probe_firing_rate_map = []

    print('I am calculating the average firing rate ...')
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[(spike_data["cluster_id"] == cluster_id)]

        trial_types = np.array(cluster_spike_data["trial_type"].tolist()[0])
        x_locations_cm = np.array(cluster_spike_data["x_position_cm"].tolist()[0])

        if len(beaconed_processed_position_data)>0:
            cluster_trial_x_locations = x_locations_cm[trial_types == 0]
            beaconed_bin_counts = np.histogram(cluster_trial_x_locations, bins)[0]
            binned_times = get_total_bin_times(beaconed_processed_position_data["times_binned"])
            normalised_rate_map = beaconed_bin_counts/binned_times
            beaconed_firing_rate_map.append(normalised_rate_map.tolist())

        if len(non_beaconed_processed_position_data)>0:
            cluster_trial_x_locations = x_locations_cm[trial_types == 1]
            non_beaconed_bin_counts = np.histogram(cluster_trial_x_locations, bins)[0]
            binned_times = get_total_bin_times(non_beaconed_processed_position_data["times_binned"])
            normalised_rate_map = non_beaconed_bin_counts/binned_times
            non_beaconed_firing_rate_map.append(normalised_rate_map.tolist())

        if len(probe_processed_position_data)>0:
            cluster_trial_x_locations = x_locations_cm[trial_types == 2]
            probe_bin_counts = np.histogram(cluster_trial_x_locations, bins)[0]
            binned_times = get_total_bin_times(probe_processed_position_data["times_binned"])
            normalised_rate_map = probe_bin_counts/binned_times
            probe_firing_rate_map.append(normalised_rate_map.tolist())
        else:
            probe_firing_rate_map.append([])
            # pass an empty list when probe trials are not present

    spike_data["beaconed_firing_rate_map"] = beaconed_firing_rate_map
    spike_data["non_beaconed_firing_rate_map"] = non_beaconed_firing_rate_map
    spike_data["probe_firing_rate_map"] = probe_firing_rate_map
    print('-------------------------------------------------------------')
    print('firing field maps processed for all trials')
    print('-------------------------------------------------------------')
    return spike_data