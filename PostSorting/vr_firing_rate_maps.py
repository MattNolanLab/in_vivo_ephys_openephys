import pandas as pd
import numpy as np



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


def bin_spikes_over_location_on_trials(spatial_data,trials,locations,number_of_trials, number_of_bins,array_of_trials):
    spike_histogram = create_2dhistogram(spatial_data,trials, locations, number_of_bins, array_of_trials)
    avg_spike_histogram = reshape_spike_histogram(spike_histogram)
    return avg_spike_histogram


def reshape_spike_histogram(spike_histogram):
    reshaped_spike_histogram = np.reshape(spike_histogram, (spike_histogram.shape[0]*spike_histogram.shape[1]))
    return reshaped_spike_histogram


def normalise_spike_number_by_time(cluster_index,spike_data,firing_rate_map, processed_position_data_dwell_time, processed_position_data):
    firing_rate_map['dwell_time'] = processed_position_data_dwell_time
    firing_rate_map['spike_rate_on_trials'] = np.nan_to_num(np.where(firing_rate_map['spike_num_on_trials'] > 0, firing_rate_map['spike_num_on_trials']/firing_rate_map['dwell_time'], 0))
    spike_data.at[cluster_index, 'avg_spike_per_bin_b'] = list(np.array(firing_rate_map['spike_rate_on_trials']))
    return spike_data


def find_spikes_on_trials(firing_rate_map, spike_data, raw_position_data, processed_position_data, cluster_index):
    bin_size_cm,number_of_bins = get_bin_size(raw_position_data) # get bin info
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    array_of_trials = np.arange(1,number_of_trials+1,1) # array of unique trial numbers

    # this calls same functions but without suming spikes over trials, saves to dataframe for further analysis in R
    firing_rate_map['spike_num_on_trials'] = bin_spikes_over_location_on_trials(raw_position_data,np.array(spike_data.at[cluster_index, 'trial_number']),np.array(spike_data.at[cluster_index, 'x_position_cm']),raw_position_data.trial_number.max(), number_of_bins,array_of_trials)
    spike_data.at[cluster_index,'spike_num_on_trials'] = firing_rate_map['spike_num_on_trials']
    return firing_rate_map,number_of_bins,array_of_trials


def make_firing_field_maps(spike_data, raw_position_data, processed_position_data, processed_position_data_binned_speed_ms_per_trial):
    print('I am calculating the average firing rate ...')
    for cluster_index in range(len(spike_data)):
        firing_rate_map = pd.DataFrame()
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        firing_rate_map,number_of_bins,array_of_trials = find_spikes_on_trials(firing_rate_map, spike_data, raw_position_data, processed_position_data, cluster_index)
        spike_data = normalise_spike_number_by_time(cluster_index,spike_data,firing_rate_map, processed_position_data_binned_speed_ms_per_trial, processed_position_data)
    print('-------------------------------------------------------------')
    print('firing field maps processed')
    print('-------------------------------------------------------------')
    return spike_data


