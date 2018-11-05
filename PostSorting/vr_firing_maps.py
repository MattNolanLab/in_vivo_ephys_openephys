import PostSorting.parameters
import numpy as np
import pandas as pd
import gc

import PostSorting.vr_spatial_data


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

"""
def average_spikes_over_trials(firing_rate_map, spike_data, spatial_data, cluster_index):
    bin_size_cm,number_of_bins = get_bin_size(spatial_data)
    beaconed_trial_no, nonbeaconed_trial_no, probe_trial_no = get_trial_numbers(spatial_data)

    avg_spikes_across_trials_b = np.zeros((len(range(int(number_of_bins)))))
    avg_spikes_across_trials_nb = np.zeros((len(range(int(number_of_bins)))))
    avg_spikes_across_trials_p = np.zeros((len(range(int(number_of_bins)))))
    for loc in range(int(number_of_bins)):
        try:
            spikes_across_trials_b=sum(firing_rate_map.loc[firing_rate_map.bin_count == loc, 'b_spike_number'])/beaconed_trial_no
            avg_spikes_across_trials_b[loc] = spikes_across_trials_b
            spikes_across_trials_nb=sum(firing_rate_map.loc[firing_rate_map.bin_count == loc, 'nb_spike_number'])/nonbeaconed_trial_no
            avg_spikes_across_trials_nb[loc] = spikes_across_trials_nb
            spikes_across_trials_p=sum(firing_rate_map.loc[firing_rate_map.bin_count == loc, 'p_spike_number'])/probe_trial_no
            avg_spikes_across_trials_p[loc] = spikes_across_trials_p
        except ZeroDivisionError: # needs fixing, not clean
            continue
    avg_spikes_across_trials_b = PostSorting.vr_spatial_data.get_rolling_sum(np.nan_to_num(avg_spikes_across_trials_b), 10)
    avg_spikes_across_trials_nb = PostSorting.vr_spatial_data.get_rolling_sum(np.nan_to_num(avg_spikes_across_trials_nb), 10)
    avg_spikes_across_trials_p = PostSorting.vr_spatial_data.get_rolling_sum(np.nan_to_num(avg_spikes_across_trials_p), 10)
    spike_data.at[cluster_index, 'avg_spike_per_bin_b'] = list(avg_spikes_across_trials_b)
    spike_data.at[cluster_index, 'avg_spike_per_bin_nb'] = list(avg_spikes_across_trials_nb)
    spike_data.at[cluster_index, 'avg_spike_per_bin_p'] = list(avg_spikes_across_trials_p)
    return spike_data


def normalise_by_time(firing_rate_map, spatial_data):
    firing_rate_map['dwell_time'] = spatial_data['binned_time_ms']
    try:
        firing_rate_map['b_spike_number'] = np.where(firing_rate_map['b_spike_number'] > 0, firing_rate_map['b_spike_number']/firing_rate_map['dwell_time'], 0)
        firing_rate_map['nb_spike_number'] = np.where(firing_rate_map['nb_spike_number'] > 0, firing_rate_map['nb_spike_number']/firing_rate_map['dwell_time'], 0)
        firing_rate_map['p_spike_number'] = np.where(firing_rate_map['p_spike_number'] > 0, firing_rate_map['p_spike_number']/firing_rate_map['dwell_time'], 0)
    except ZeroDivisionError or ValueError:
        return firing_rate_map
    return firing_rate_map


def find_spikes_on_trials(firing_rate_map, spike_data, spatial_data, cluster_index):
    bin_size_cm,number_of_bins = get_bin_size(spatial_data)
    number_of_trials = spatial_data.trial_number.max() # total number of trials
    trials_b = np.array(spike_data.at[cluster_index, 'beaconed_trial_number']);locations_b = np.array(spike_data.at[cluster_index, 'beaconed_position_cm'])
    trials_nb = np.array(spike_data.at[cluster_index,'nonbeaconed_trial_number']);locations_nb = np.array(spike_data.at[cluster_index, 'nonbeaconed_position_cm'])
    trials_p = np.array(spike_data.at[cluster_index, 'probe_trial_number']);locations_p = np.array(spike_data.at[cluster_index, 'probe_position_cm'])
    for t in range(1,int(number_of_trials)):
        try:
            trial_locations_b = np.take(locations_b, np.where(trials_b == t)[0])
            trial_locations_nb = np.take(locations_nb, np.where(trials_nb == t)[0])
            trial_locations_p = np.take(locations_p, np.where(trials_p == t)[0])
            for loc in range(int(number_of_bins)):
                spikes_in_bin_b = trial_locations_b[np.where(np.logical_and(trial_locations_b > loc, trial_locations_b <= (loc+1)))]
                spikes_in_bin_nb = trial_locations_nb[np.where(np.logical_and(trial_locations_nb > loc, trial_locations_nb <= (loc+1)))]
                spikes_in_bin_p = trial_locations_p[np.where(np.logical_and(trial_locations_p > loc, trial_locations_p <= (loc+1)))]
                firing_rate_map = firing_rate_map.append({"trial_number": int(t), "bin_count": int(loc), "b_spike_number":  len(spikes_in_bin_b), "nb_spike_number":  len(spikes_in_bin_nb), "p_spike_number":  len(spikes_in_bin_p)}, ignore_index=True)
        except IndexError: # if there is no spikes on that trial /// # needs fixing, not clean
            for loc in range(int(number_of_bins)):
                firing_rate_map = firing_rate_map.append({"trial_number": int(t),"bin_count": int(loc),"b_spike_number":  0, "nb_spike_number":  0, "p_spike_number":  0}, ignore_index=True)
    return firing_rate_map
"""

def reshape_and_sum_binned_normalised_spikes(normalised_spikes, number_of_trials, number_of_bins, array_of_trials):
    reshaped_normalised_spikes = np.reshape(normalised_spikes, (len(array_of_trials),int(number_of_bins)))
    average_spikes_over_trials = np.sum(reshaped_normalised_spikes, axis = 0)/number_of_trials
    return average_spikes_over_trials


def average_normalised_spikes_over_trials(firing_rate_map, spike_data, spatial_data, cluster_index,number_of_bins,array_of_trials):
    beaconed_normalised_spikes = np.array(firing_rate_map['normalised_b_spike_number'])
    nonbeaconed_normalised_spikes = np.array(firing_rate_map['normalised_nb_spike_number'])
    probe_normalised_spikes = np.array(firing_rate_map['normalised_p_spike_number'])

    number_of_beaconed_trials,number_of_nonbeaconed_trials, number_of_probe_trials = get_trial_numbers(spatial_data)

    average_spikes_over_trials = reshape_and_sum_binned_normalised_spikes(beaconed_normalised_spikes, number_of_beaconed_trials, number_of_bins,array_of_trials)
    average_spikes_over_trials = PostSorting.vr_spatial_data.get_rolling_sum(np.nan_to_num(average_spikes_over_trials), 10)
    spike_data.at[cluster_index, 'avg_spike_per_bin_b'] = list(average_spikes_over_trials)
    average_spikes_over_trials = reshape_and_sum_binned_normalised_spikes(nonbeaconed_normalised_spikes, number_of_nonbeaconed_trials, number_of_bins,array_of_trials)
    average_spikes_over_trials = PostSorting.vr_spatial_data.get_rolling_sum(np.nan_to_num(average_spikes_over_trials), 10)
    spike_data.at[cluster_index, 'avg_spike_per_bin_nb'] = list(average_spikes_over_trials)
    average_spikes_over_trials = reshape_and_sum_binned_normalised_spikes(probe_normalised_spikes, number_of_probe_trials, number_of_bins,array_of_trials)
    average_spikes_over_trials = PostSorting.vr_spatial_data.get_rolling_sum(np.nan_to_num(average_spikes_over_trials), 10)
    spike_data.at[cluster_index, 'avg_spike_per_bin_p'] = list(average_spikes_over_trials)
    return spike_data


def normalise_spike_number_by_time(spatial_data,firing_rate_map):
    firing_rate_map['dwell_time'] = spatial_data['binned_time_ms']
    firing_rate_map['normalised_b_spike_number'] = np.where(firing_rate_map['b_spike_number'] > 0, firing_rate_map['b_spike_number']/firing_rate_map['dwell_time'], 0)
    firing_rate_map['normalised_nb_spike_number'] = np.where(firing_rate_map['nb_spike_number'] > 0, firing_rate_map['nb_spike_number']/firing_rate_map['dwell_time'], 0)
    firing_rate_map['normalised_p_spike_number'] = np.where(firing_rate_map['p_spike_number'] > 0, firing_rate_map['p_spike_number']/firing_rate_map['dwell_time'], 0)
    return firing_rate_map


def bin_spikes(spatial_data,trials,locations,number_of_trials, number_of_bins,array_of_trials):
    spike_histogram = create_2dhistogram(spatial_data,trials, locations, number_of_bins, array_of_trials)
    shape_of_array = number_of_trials*number_of_bins
    reshaped_spike_histogram = np.ravel(np.reshape(spike_histogram, (int(shape_of_array),1), order='C'))
    return reshaped_spike_histogram


def find_spikes_on_trials(firing_rate_map, spike_data, spatial_data, cluster_index):
    bin_size_cm,number_of_bins = get_bin_size(spatial_data)
    trials_b = np.array(spike_data.at[cluster_index, 'beaconed_trial_number']);locations_b = np.array(spike_data.at[cluster_index, 'beaconed_position_cm'])
    trials_nb = np.array(spike_data.at[cluster_index,'nonbeaconed_trial_number']);locations_nb = np.array(spike_data.at[cluster_index, 'nonbeaconed_position_cm'])
    trials_p = np.array(spike_data.at[cluster_index, 'probe_trial_number']);locations_p = np.array(spike_data.at[cluster_index, 'probe_position_cm'])

    number_of_trials = spatial_data.trial_number.max() # total number of trials
    array_of_trials = np.arange(1,number_of_trials+1,1)

    firing_rate_map['b_spike_number'] = bin_spikes(spatial_data,trials_b,locations_b,number_of_trials, number_of_bins,array_of_trials)
    firing_rate_map['nb_spike_number'] = bin_spikes(spatial_data,trials_nb,locations_nb,number_of_trials, number_of_bins,array_of_trials)
    firing_rate_map['p_spike_number'] = bin_spikes(spatial_data,trials_p,locations_p,number_of_trials, number_of_bins,array_of_trials)
    return firing_rate_map,number_of_bins,array_of_trials


def average_spikes_across_trials(spike_data,spatial_data,cluster_index):
    bin_size_cm,number_of_bins = get_bin_size(spatial_data)
    number_of_beaconed_trials,number_of_nonbeaconed_trials, number_of_probe_trials = get_trial_numbers(spatial_data)
    trials_b = np.array(spike_data.at[cluster_index, 'beaconed_trial_number']);locations_b = np.array(spike_data.at[cluster_index, 'beaconed_position_cm'])

    posrange = np.linspace(spatial_data.x_position_cm.min(), spatial_data.x_position_cm.max(), num=number_of_bins+1)
    spike_histogram, bins = np.histogram(trials_b, locations_b, bins=(posrange), range=None)

    number_of_trials = spatial_data.trial_number.max() # total number of trials
    array_of_trials = np.arange(1,number_of_trials+1,1)
    binned_time_ms = np.array(spatial_data['binned_time_ms'])
    reshaped_binned_time_ms = np.reshape(binned_time_ms, (len(array_of_trials),int(number_of_bins)))
    average_reshaped_binned_time_ms = np.sum(reshaped_binned_time_ms, axis = 0)/number_of_trials
    normalised_spikes = spike_histogram/average_reshaped_binned_time_ms
    divide_by_trials = normalised_spikes/number_of_trials

    return spike_data


def make_firing_field_maps(spike_data, spatial_data, prm):
    print('I am calculating the average firing rate ...')
    gc.collect()
    for cluster_index in range(len(spike_data)):
        firing_rate_map = pd.DataFrame()
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1

        firing_rate_map,number_of_bins,array_of_trials = find_spikes_on_trials(firing_rate_map, spike_data, spatial_data, cluster_index)
        firing_rate_map = normalise_spike_number_by_time(spatial_data,firing_rate_map)
        spike_data = average_normalised_spikes_over_trials(firing_rate_map, spike_data, spatial_data, cluster_index,number_of_bins,array_of_trials)
    return spike_data

