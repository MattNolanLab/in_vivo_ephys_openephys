import PostSorting.parameters
import numpy as np
import pandas as pd
import settings
from astropy.convolution import convolve, Gaussian1DKernel

prm = PostSorting.parameters.Parameters()

def add_speed(spike_data, raw_position_data):
    speed_per200ms = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        speed_per200ms.append(raw_position_data["speed_per200ms"][cluster_firing_indices].to_list())

    spike_data["speed_per200ms"] = speed_per200ms
    return spike_data


def add_position_x(spike_data, raw_position_data):
    x_position_cm = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        x_position_cm.append(raw_position_data["x_position_cm"][cluster_firing_indices].to_list())

    spike_data["x_position_cm"] = x_position_cm
    return spike_data


def add_trial_number(spike_data, raw_position_data):
    trial_number = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        trial_number.append(raw_position_data["trial_number"][cluster_firing_indices].to_list())

    spike_data["trial_number"] = trial_number
    return spike_data


def add_trial_type(spike_data, raw_position_data):
    trial_type = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        trial_type.append(raw_position_data["trial_type"][cluster_firing_indices].to_list())

    spike_data["trial_type"] = trial_type
    return spike_data

def bin_fr_in_time(spike_data, raw_position_data):
    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_time_seconds/settings.time_bin_size)

    # make an empty list of list for all firing rates binned in time for each cluster
    fr_binned_in_time = [[] for x in range(len(spike_data))]

    for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
        trial_times = np.array(raw_position_data['time_seconds'][np.array(raw_position_data['trial_number']) == trial_number], dtype="float64")
        time_bins = np.arange(min(trial_times), max(trial_times), settings.time_bin_size)# 100ms time bins

        for i, cluster_id in enumerate(spike_data.cluster_id):
            if len(time_bins)>1:
                firing_times = spike_data[spike_data["cluster_id"] == cluster_id]["firing_times"].iloc[0]
                trial_numbers = np.array(spike_data[spike_data["cluster_id"] == cluster_id]["trial_number"].iloc[0])
                trial_firing_times = firing_times[trial_numbers == trial_number]

                # convert spike indices to spike times, count the spikes in each time bin and divide by time bin size to get fr
                fr_hist = np.histogram(trial_firing_times/settings.sampling_rate, time_bins)[0]/settings.time_bin_size

                # and smooth
                fr_hist = convolve(fr_hist, gauss_kernel)

                # append the binned firing for each cluster at each trial
                fr_binned_in_time[i].append(fr_hist.tolist())
            else:
                fr_binned_in_time[i].append([])

    spike_data["fr_time_binned"] = fr_binned_in_time

    return spike_data


def bin_fr_in_space(spike_data, raw_position_data, track_length):
    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/settings.vr_bin_size_cm)

    # make an empty list of list for all firing rates binned in time for each cluster
    fr_binned_in_space = [[] for x in range(len(spike_data))]
    fr_binned_in_space_bin_centres = [[] for x in range(len(spike_data))]

    for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
        trial_x_position_cm = np.array(raw_position_data['x_position_cm'][np.array(raw_position_data['trial_number']) == trial_number], dtype="float64")
        trial_x_dwell_time = np.array(raw_position_data['dwell_time_ms'][np.array(raw_position_data['trial_number']) == trial_number], dtype="float64")
        pos_bins = np.arange(0, track_length, settings.vr_bin_size_cm)# 100ms time bins

        for i, cluster_id in enumerate(spike_data.cluster_id):
            if len(pos_bins)>1:
                x_position_cm = np.array(spike_data[spike_data["cluster_id"] == cluster_id]["x_position_cm"].iloc[0])
                trial_numbers = np.array(spike_data[spike_data["cluster_id"] == cluster_id]["trial_number"].iloc[0])
                trial_firing_locations = x_position_cm[trial_numbers == trial_number]

                # count the spikes in each space bin and normalise by the total time spent in that bin for the trial
                fr_hist, bin_edges = np.histogram(trial_firing_locations, pos_bins)
                fr_hist = fr_hist/(np.histogram(trial_x_position_cm, pos_bins, weights=trial_x_dwell_time)[0])

                # get location bin centres
                bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

                # nans to zero and smooth
                fr_hist[np.isnan(fr_hist)] = 0
                fr_hist = convolve(fr_hist, gauss_kernel)

                # append the binned firing for each cluster at each trial
                fr_binned_in_space[i].append(fr_hist.tolist())
                fr_binned_in_space_bin_centres[i].append(bin_centres.tolist())
            else:
                fr_binned_in_space[i].append([])
                fr_binned_in_space_bin_centres[i].append([])

    spike_data["fr_binned_in_space"] = fr_binned_in_space
    spike_data["fr_binned_in_space_bin_centres"] = fr_binned_in_space_bin_centres

    return spike_data


def add_location_and_task_variables(spike_data, raw_position_data, track_length):
    print('I am extracting firing locations for each cluster...')
    spike_data = add_speed(spike_data, raw_position_data)
    spike_data = add_position_x(spike_data, raw_position_data)
    spike_data = add_trial_number(spike_data, raw_position_data)
    spike_data = add_trial_type(spike_data, raw_position_data)

    spike_data = bin_fr_in_time(spike_data, raw_position_data)
    spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length)
    return spike_data


def split_and_add_trial_number(cluster_index, spike_data_movement, spike_data_stationary, spike_data_trial_number,above_threshold_indices,below_threshold_indices):
    spike_data_movement.trial_number.iloc[cluster_index] = spike_data_trial_number[above_threshold_indices]
    spike_data_stationary.trial_number.iloc[cluster_index] = spike_data_trial_number[below_threshold_indices]
    return spike_data_movement, spike_data_stationary


def split_and_add_x_location_cm(cluster_index, spike_data_movement, spike_data_stationary, spike_data_x_location_cm,above_threshold_indices,below_threshold_indices):
    spike_data_movement.x_position_cm.iloc[cluster_index] = spike_data_x_location_cm[above_threshold_indices]
    spike_data_stationary.x_position_cm.iloc[cluster_index] = spike_data_x_location_cm[below_threshold_indices]
    return spike_data_movement, spike_data_stationary


def split_and_add_trial_type(cluster_index, spike_data_movement, spike_data_stationary, spike_data_trial_type,above_threshold_indices,below_threshold_indices):
    spike_data_movement.trial_type.iloc[cluster_index] = spike_data_trial_type[above_threshold_indices]
    spike_data_stationary.trial_type.iloc[cluster_index] = spike_data_trial_type[below_threshold_indices]
    return spike_data_movement, spike_data_stationary


def split_spatial_firing_by_speed(spike_data, spike_data_movement, spike_data_stationary):
    movement_threshold=settings.movement_threshold # 2.5 cm / second

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        above_threshold_indices = np.where(np.array(spike_data.speed_per200ms.iloc[cluster_index]) >= movement_threshold)[0]
        below_threshold_indices = np.where(np.array(spike_data.speed_per200ms.iloc[cluster_index]) < movement_threshold)[0]

        spike_data_movement, spike_data_stationary = split_and_add_trial_number(cluster_index, spike_data_movement, spike_data_stationary, np.array(spike_data.trial_number.iloc[cluster_index]), above_threshold_indices, below_threshold_indices)
        spike_data_movement, spike_data_stationary = split_and_add_x_location_cm(cluster_index,spike_data_movement, spike_data_stationary, np.array(spike_data.x_position_cm.iloc[cluster_index]),above_threshold_indices, below_threshold_indices)
        spike_data_movement, spike_data_stationary = split_and_add_trial_type(cluster_index,   spike_data_movement, spike_data_stationary, np.array(spike_data.trial_type.iloc[cluster_index]),   above_threshold_indices, below_threshold_indices)
    return spike_data_movement, spike_data_stationary


def split_spatial_firing_by_trial_type(spike_data):
    print('I am splitting firing locations by trial type...')
    beaconed_position_cm = []
    beaconed_trial_number = []
    nonbeaconed_position_cm = []
    nonbeaconed_trial_number = []
    probe_position_cm = []
    probe_trial_number = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        trials = np.array(cluster_df['trial_number'].tolist())
        locations = np.array(cluster_df['x_position_cm'].tolist())
        trial_type = np.array(cluster_df['trial_type'].tolist())

        beaconed_position_cm.append(locations[trial_type == 0])
        beaconed_trial_number.append(trials[trial_type == 0])
        nonbeaconed_position_cm.append(locations[trial_type == 1])
        nonbeaconed_trial_number.append(trials[trial_type == 1])
        probe_position_cm.append(locations[trial_type == 2])
        probe_trial_number.append(trials[trial_type == 2])

    spike_data["beaconed_position_cm"] = beaconed_position_cm
    spike_data["beaconed_trial_number"] = beaconed_trial_number
    spike_data["nonbeaconed_position_cm"] = nonbeaconed_position_cm
    spike_data["nonbeaconed_trial_number"] = nonbeaconed_trial_number
    spike_data["probe_position_cm"] = probe_position_cm
    spike_data["probe_trial_number"] = probe_trial_number
    return spike_data


def process_spatial_firing(spike_data, raw_position_data, track_length):
    spike_data_movement = spike_data.copy()
    spike_data_stationary = spike_data.copy()

    spike_data = add_location_and_task_variables(spike_data, raw_position_data, track_length)
    spike_data = split_spatial_firing_by_trial_type(spike_data)
    print('-------------------------------------------------------------')
    print('spatial firing processed')
    print('-------------------------------------------------------------')
    return spike_data_movement, spike_data_stationary, spike_data

