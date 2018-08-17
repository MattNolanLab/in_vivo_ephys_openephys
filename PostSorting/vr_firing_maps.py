import PostSorting.parameters
import numpy as np
import pandas as pd

prm = PostSorting.parameters.Parameters()


def calculate_dwell_time(spatial_data):
    bin_size_cm,number_of_bins = get_bin_size(spatial_data,prm)
    number_of_trials = spatial_data.trial_number.max() # total number of trials

    for loc in range(int(number_of_bins)):
        across_trials = spatial_data.loc[lambda spatial_data: spatial_data.bin_count == loc, 'dwell_time_ms'].mean()


def get_bin_size(spatial_data):
    bin_size_cm = 5
    track_length = spatial_data.position_cm.max()
    start_of_track = spatial_data.position_cm.min()
    number_of_bins = (track_length - start_of_track)/bin_size_cm
    return bin_size_cm,number_of_bins


def average_spikes_over_trials(firing_rate_map,number_of_bins):
    avg_spikes_across_trials = np.zeros((len(range(int(number_of_bins)))))
    number_of_trials = firing_rate_map.trial_number.max() # total number of trials
    for loc in range(int(number_of_bins)):
        spikes_across_trials=firing_rate_map.loc[firing_rate_map.bin_count == loc, 'spike_number'].sum()/int(number_of_trials)
        avg_spikes_across_trials[loc] = spikes_across_trials

    return avg_spikes_across_trials


def calculate_firing_rate(spike_data, spatial_data):
    print('I am calculating the firing rate map...')

    firing_rate_map = pd.DataFrame(columns=['trial_number', 'bin_count', 'spike_number', 'dwell_time_ms'])

    cluster = 0 # for testing, plot one cluster
    bin_size_cm,number_of_bins = get_bin_size(spatial_data)
    number_of_trials = spatial_data.trial_number.max() # total number of trials
    cluster_df = spike_data.iloc[[cluster]] # dataframe for that cluster
    trials = np.array(cluster_df.trial_number.tolist())
    locations = np.array(cluster_df.position_cm.tolist())
    for t in range(1,int(number_of_trials)):
        surplus,trial_indices=np.where(trials == t)
        trial_locations = np.take(locations, trial_indices)
        for loc in range(int(number_of_bins)):
            min_loc_cm = 5*loc
            max_loc_cm = 5*(loc+1)
            spikes_in_bin = trial_locations[np.where(np.logical_and(trial_locations > min_loc_cm, trial_locations < max_loc_cm))]

            firing_rate_map = firing_rate_map.append({
                "trial_number": int(t),
                "bin_count": int(loc),
                "spike_number":  len(spikes_in_bin),
            }, ignore_index=True)


    avg_spike_per_bin = average_spikes_over_trials(firing_rate_map,number_of_bins)

    spike_data = spike_data.append({"avg_spike_per_bin": list(avg_spike_per_bin)}, ignore_index=True)

    print('firing rate map has been calculated')

    return spike_data


def make_firing_field_maps(spike_data, spatial_data):

    spike_data = calculate_firing_rate(spike_data, spatial_data)

    return spike_data