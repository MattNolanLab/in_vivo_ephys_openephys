import pandas as pd
import numpy as np


def get_trial_numbers(spatial_data):
    beaconed_trial_no = spatial_data.at[0,'beaconed_total_trial_number']
    nonbeaconed_trial_no = spatial_data.at[0,'nonbeaconed_total_trial_number']
    probe_trial_no = spatial_data.at[0,'probe_total_trial_number']
    return beaconed_trial_no, nonbeaconed_trial_no, probe_trial_no


def get_bin_size(spatial_data):
    bin_size_cm = 1
    track_length = spatial_data.x_position_cm.max()
    start_of_track = spatial_data.x_position_cm.min()
    number_of_bins = (track_length - start_of_track)/bin_size_cm
    return bin_size_cm,number_of_bins


def create_2dhistogram(spatial_data,trials, locations, number_of_bins, array_of_trials):
    posrange = np.linspace(spatial_data.x_position_cm.min(), spatial_data.x_position_cm.max(), num=number_of_bins+1)
    trialrange = np.unique(array_of_trials)
    trialrange = np.append(trialrange, trialrange[-1]+1)  # Add end of range
    values = np.array([[trialrange[0], trialrange[-1]],[posrange[0], posrange[-1]]])
    H, bins, ranges = np.histogram2d(trials, locations, bins=(trialrange, posrange), range=values)
    return H


def bin_spikes_over_location_on_trials(spatial_data,trials,locations, number_of_bins,array_of_trials):
    spike_histogram = create_2dhistogram(spatial_data,trials, locations, number_of_bins, array_of_trials)
    avg_spike_histogram = reshape_spike_histogram(spike_histogram)
    return avg_spike_histogram


def reshape_spike_histogram(spike_histogram):
    reshaped_spike_histogram = np.reshape(spike_histogram, (spike_histogram.shape[0]*spike_histogram.shape[1]))
    return reshaped_spike_histogram


def reshape_to_average_over_trials(array, number_of_trials, number_of_trial_type_trials):
    reshaped_spike_histogram = np.reshape(array, (200, int(number_of_trials)))
    avg_spike_histogram = np.sum(reshaped_spike_histogram, axis=1)/number_of_trial_type_trials
    return avg_spike_histogram


def average_over_trials(cluster_index, spike_data, number_of_trials, processed_position_data):
    number_of_beaconed_trials,number_of_nonbeaconed_trials, number_of_probe_trials = get_trial_numbers(processed_position_data)
    reshaped_spike_histogram = reshape_to_average_over_trials(np.array(spike_data.at[cluster_index, 'b_spike_rate_on_trials']), number_of_trials, number_of_beaconed_trials)
    spike_data.at[cluster_index, 'avg_b_spike_rate'] = list(reshaped_spike_histogram)
    reshaped_spike_histogram = reshape_to_average_over_trials(np.array(spike_data.at[cluster_index, 'nb_spike_rate_on_trials']), number_of_trials, number_of_nonbeaconed_trials)
    spike_data.at[cluster_index, 'avg_nb_spike_rate'] = list(reshaped_spike_histogram)
    reshaped_spike_histogram = reshape_to_average_over_trials(np.array(spike_data.at[cluster_index, 'p_spike_rate_on_trials']), number_of_trials, number_of_probe_trials)
    spike_data.at[cluster_index, 'avg_p_spike_rate'] = list(reshaped_spike_histogram)
    return spike_data


def normalise_spike_number_by_time(cluster_index,spike_data,firing_rate_map, processed_position_data_dwell_time):
    firing_rate_map['dwell_time'] = processed_position_data_dwell_time
    firing_rate_map['b_spike_rate_on_trials'] = np.nan_to_num(np.where(firing_rate_map['b_spike_num_on_trials'] > 0, firing_rate_map['b_spike_num_on_trials']/firing_rate_map['dwell_time'], 0))
    spike_data.at[cluster_index, 'b_spike_rate_on_trials'] = list(np.array(firing_rate_map['b_spike_rate_on_trials']))
    firing_rate_map['nb_spike_rate_on_trials'] = np.nan_to_num(np.where(firing_rate_map['nb_spike_num_on_trials'] > 0, firing_rate_map['nb_spike_num_on_trials']/firing_rate_map['dwell_time'], 0))
    spike_data.at[cluster_index, 'nb_spike_rate_on_trials'] = list(np.array(firing_rate_map['nb_spike_rate_on_trials']))
    firing_rate_map['p_spike_rate_on_trials'] = np.nan_to_num(np.where(firing_rate_map['p_spike_num_on_trials'] > 0, firing_rate_map['p_spike_num_on_trials']/firing_rate_map['dwell_time'], 0))
    spike_data.at[cluster_index, 'p_spike_rate_on_trials'] = list(np.array(firing_rate_map['p_spike_rate_on_trials']))
    return spike_data


def find_spikes_on_trials(firing_rate_map, spike_data, raw_position_data, cluster_index):
    bin_size_cm,number_of_bins = get_bin_size(raw_position_data) # get bin info
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    array_of_trials = np.arange(1,number_of_trials+1,1) # array of unique trial numbers
    firing_rate_map['b_spike_num_on_trials'] = bin_spikes_over_location_on_trials(raw_position_data,np.array(spike_data.at[cluster_index, 'beaconed_trial_number']), np.array(spike_data.at[cluster_index, 'beaconed_position_cm']), number_of_bins,array_of_trials)
    spike_data.at[cluster_index,'b_spike_num_on_trials'] = list(np.array(firing_rate_map['b_spike_num_on_trials']))
    firing_rate_map['nb_spike_num_on_trials'] = bin_spikes_over_location_on_trials(raw_position_data,np.array(spike_data.at[cluster_index, 'nonbeaconed_trial_number']), np.array(spike_data.at[cluster_index, 'nonbeaconed_position_cm']), number_of_bins,array_of_trials)
    spike_data.at[cluster_index,'nb_spike_num_on_trials'] = list(np.array(firing_rate_map['nb_spike_num_on_trials']))
    firing_rate_map['p_spike_num_on_trials'] = bin_spikes_over_location_on_trials(raw_position_data,np.array(spike_data.at[cluster_index, 'probe_trial_number']), np.array(spike_data.at[cluster_index, 'probe_position_cm']), number_of_bins,array_of_trials)
    spike_data.at[cluster_index,'p_spike_num_on_trials'] = list(np.array(firing_rate_map['p_spike_num_on_trials']))
    return firing_rate_map,number_of_bins,array_of_trials,spike_data


def make_firing_field_maps(spike_data, raw_position_data, processed_position_data):
    print('I am calculating the average firing rate ...')
    for cluster_index in range(len(spike_data)):
        firing_rate_map = pd.DataFrame()
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        firing_rate_map,number_of_bins,array_of_trials,spike_data = find_spikes_on_trials(firing_rate_map, spike_data, raw_position_data, cluster_index)
        spike_data = normalise_spike_number_by_time(cluster_index,spike_data,firing_rate_map, processed_position_data.binned_time_ms_per_trial)
        spike_data = average_over_trials(cluster_index,spike_data, raw_position_data.trial_number.max(), processed_position_data)
        #smooth_firing_rate(cluster_index,spike_data)
    print('-------------------------------------------------------------')
    print('firing field maps processed')
    print('-------------------------------------------------------------')
    return spike_data


