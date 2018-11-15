import PostSorting.parameters
import numpy as np
import pandas as pd
import gc

import PostSorting.vr_sync_spatial_data


def get_bin_size(spatial_data):
    bin_size_cm = 1
    track_length = spatial_data.x_position_cm.max()
    start_of_track = spatial_data.x_position_cm.min()
    number_of_bins = (track_length - start_of_track)/bin_size_cm
    return bin_size_cm,number_of_bins

#@jit
def gaussian_kernel(kernx):
    kerny = np.exp(np.power(kernx, 2)/2 * (-1))
    return kerny


def round_down(num, divisor):
    return num - (num%divisor)


def create_2dhistogram(spatial_data,trials, locations, number_of_bins, array_of_trials):
    posrange = np.linspace(spatial_data.x_position_cm.min(), spatial_data.x_position_cm.max(), num=number_of_bins+1)
    trialrange = np.unique(array_of_trials)
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],[posrange[0], posrange[-1]]])

    H, bins, ranges = np.histogram2d(trials, locations, bins=(trialrange, posrange), range=values)
    return H


def get_trial_numbers(spatial_data):
    beaconed_trial_no = spatial_data.at[0,'beaconed_total_trial_number']
    nonbeaconed_trial_no = spatial_data.at[0,'nonbeaconed_total_trial_number']
    probe_trial_no = spatial_data.at[0,'probe_total_trial_number']
    return beaconed_trial_no, nonbeaconed_trial_no, probe_trial_no


def reshape_and_sum_binned_normalised_spikes(normalised_spikes, number_of_trials, number_of_bins, array_of_trials):
    reshaped_normalised_spikes = np.reshape(normalised_spikes, (len(array_of_trials),int(number_of_bins)))
    average_spikes_over_trials = np.sum(reshaped_normalised_spikes, axis = 0)/number_of_trials
    return average_spikes_over_trials


def average_normalised_spikes_over_trials(firing_rate_map, spike_data, processed_position_data, cluster_index,number_of_bins,array_of_trials):
    beaconed_normalised_spikes = np.array(firing_rate_map['normalised_b_spike_number'])
    nonbeaconed_normalised_spikes = np.array(firing_rate_map['normalised_nb_spike_number'])
    probe_normalised_spikes = np.array(firing_rate_map['normalised_p_spike_number'])

    spike_data.at[cluster_index, 'normalised_b_spike_number'] = list(beaconed_normalised_spikes)
    spike_data.at[cluster_index, 'normalised_nb_spike_number'] = list(nonbeaconed_normalised_spikes)
    spike_data.at[cluster_index, 'normalised_p_spike_number'] = list(probe_normalised_spikes)

    number_of_beaconed_trials,number_of_nonbeaconed_trials, number_of_probe_trials = get_trial_numbers(processed_position_data)

    average_spikes_over_trials = reshape_and_sum_binned_normalised_spikes(beaconed_normalised_spikes, number_of_beaconed_trials, number_of_bins,array_of_trials)
    average_spikes_over_trials = PostSorting.vr_sync_spatial_data.get_rolling_sum(np.nan_to_num(average_spikes_over_trials), 10)
    spike_data.at[cluster_index, 'avg_spike_per_bin_b'] = list(average_spikes_over_trials)
    average_spikes_over_trials = reshape_and_sum_binned_normalised_spikes(nonbeaconed_normalised_spikes, number_of_nonbeaconed_trials, number_of_bins,array_of_trials)
    average_spikes_over_trials = PostSorting.vr_sync_spatial_data.get_rolling_sum(np.nan_to_num(average_spikes_over_trials), 10)
    spike_data.at[cluster_index, 'avg_spike_per_bin_nb'] = list(average_spikes_over_trials)
    average_spikes_over_trials = reshape_and_sum_binned_normalised_spikes(probe_normalised_spikes, number_of_probe_trials, number_of_bins,array_of_trials)
    average_spikes_over_trials = PostSorting.vr_sync_spatial_data.get_rolling_sum(np.nan_to_num(average_spikes_over_trials), 10)
    spike_data.at[cluster_index, 'avg_spike_per_bin_p'] = list(average_spikes_over_trials)
    return spike_data


def normalise_spike_number_by_time(processed_position_data,firing_rate_map):
    firing_rate_map['dwell_time'] = processed_position_data['binned_time_ms']
    firing_rate_map['normalised_b_spike_number'] = np.where(firing_rate_map['b_spike_number'] > 0, firing_rate_map['b_spike_number']/firing_rate_map['dwell_time'], 0)
    firing_rate_map['normalised_nb_spike_number'] = np.where(firing_rate_map['nb_spike_number'] > 0, firing_rate_map['nb_spike_number']/firing_rate_map['dwell_time'], 0)
    firing_rate_map['normalised_p_spike_number'] = np.where(firing_rate_map['p_spike_number'] > 0, firing_rate_map['p_spike_number']/firing_rate_map['dwell_time'], 0)
    return firing_rate_map


def bin_spikes(spatial_data,trials,locations,number_of_trials, number_of_bins,array_of_trials):
    spike_histogram = create_2dhistogram(spatial_data,trials, locations, number_of_bins, array_of_trials)
    shape_of_array = number_of_trials*number_of_bins
    reshaped_spike_histogram = np.ravel(np.reshape(spike_histogram, (int(shape_of_array),1), order='C'))
    return reshaped_spike_histogram


def find_spikes_on_trials(firing_rate_map, spike_data, raw_position_data, processed_position_data, cluster_index):
    bin_size_cm,number_of_bins = get_bin_size(raw_position_data)
    trials_b = np.array(spike_data.at[cluster_index, 'beaconed_trial_number']);locations_b = np.array(spike_data.at[cluster_index, 'beaconed_position_cm'])
    trials_nb = np.array(spike_data.at[cluster_index,'nonbeaconed_trial_number']);locations_nb = np.array(spike_data.at[cluster_index, 'nonbeaconed_position_cm'])
    trials_p = np.array(spike_data.at[cluster_index, 'probe_trial_number']);locations_p = np.array(spike_data.at[cluster_index, 'probe_position_cm'])

    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    array_of_trials = np.arange(1,number_of_trials+1,1)

    firing_rate_map['b_spike_number'] = bin_spikes(raw_position_data,trials_b,locations_b,number_of_trials, number_of_bins,array_of_trials)
    firing_rate_map['nb_spike_number'] = bin_spikes(raw_position_data,trials_nb,locations_nb,number_of_trials, number_of_bins,array_of_trials)
    firing_rate_map['p_spike_number'] = bin_spikes(raw_position_data,trials_p,locations_p,number_of_trials, number_of_bins,array_of_trials)
    return firing_rate_map,number_of_bins,array_of_trials


def average_spikes_across_trials(spike_data,spatial_data,cluster_index):
    bin_size_cm,number_of_bins = get_bin_size(spatial_data)
    number_of_beaconed_trials,number_of_nonbeaconed_trials, number_of_probe_trials = get_trial_numbers(spatial_data)
    trials_b = np.array(spike_data.at[cluster_index, 'beaconed_trial_number']);locations_b = np.array(spike_data.at[cluster_index, 'beaconed_position_cm'])

    posrange = np.linspace(spatial_data.x_position_cm.min(), spatial_data.x_position_cm.max(), num=number_of_bins+1)
    spike_histogram, bins = np.histogram(trials_b, locations_b, bins=(posrange), range=None)

    number_of_trials = spatial_data.trial_number.max() # total number of trials
    array_of_trials = np.arange(1,number_of_trials+1,1)
    binned_time_ms = np.array(spatial_data['binned_time_over_trials_seconds'])
    normalised_spikes = spike_histogram/binned_time_ms
    divide_by_trials = normalised_spikes/number_of_trials

    return spike_data


def make_firing_field_maps(spike_data, raw_position_data, processed_position_data, prm):
    print('I am calculating the average firing rate ...')
    gc.collect()
    for cluster_index in range(len(spike_data)):
        firing_rate_map = pd.DataFrame()
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1

        firing_rate_map,number_of_bins,array_of_trials = find_spikes_on_trials(firing_rate_map, spike_data, raw_position_data, processed_position_data, cluster_index)
        firing_rate_map = normalise_spike_number_by_time(processed_position_data,firing_rate_map)
        spike_data = average_normalised_spikes_over_trials(firing_rate_map, spike_data, processed_position_data, cluster_index,number_of_bins,array_of_trials)
    return spike_data,firing_rate_map

