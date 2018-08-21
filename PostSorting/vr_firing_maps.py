import PostSorting.parameters
import numpy as np
import pandas as pd

prm = PostSorting.parameters.Parameters()


def calculate_dwell_time(spatial_data):
    bin_size_cm,number_of_bins = get_bin_size(spatial_data,prm)
    number_of_trials = spatial_data.trial_number.max() # total number of trials

    for t in range(1,int(number_of_trials)):
        for loc in range(int(number_of_bins)):
            dwell_time = spatial_data.loc[lambda spatial_data: spatial_data.bin_count == loc, 'dwell_time_ms'].mean()

            firing_rate_map = firing_rate_map.append({
                "dwell_time_ms": dwell_time,
            }, ignore_index=True)


def get_bin_size(spatial_data):
    bin_size_cm = 5
    track_length = spatial_data.position_cm.max()
    start_of_track = spatial_data.position_cm.min()
    number_of_bins = (track_length - start_of_track)/bin_size_cm
    return bin_size_cm,number_of_bins


def average_spikes_over_trials(firing_rate_map,number_of_bins):
    avg_spikes_across_trials_b = np.zeros((len(range(int(number_of_bins)))))
    avg_spikes_across_trials_nb = np.zeros((len(range(int(number_of_bins)))))
    number_of_trials = firing_rate_map.trial_number.max() # total number of trials
    for loc in range(int(number_of_bins)):
        spikes_across_trials=firing_rate_map.loc[firing_rate_map.bin_count == loc, 'b_spike_number'].sum()/int(number_of_trials)
        avg_spikes_across_trials_b[loc] = spikes_across_trials
        spikes_across_trials=firing_rate_map.loc[firing_rate_map.bin_count == loc, 'nb_spike_number'].sum()/int(number_of_trials)
        avg_spikes_across_trials_nb[loc] = spikes_across_trials
    return avg_spikes_across_trials_b,avg_spikes_across_trials_nb


def calculate_firing_rate(spike_data, spatial_data):
    print('I am calculating the average firing rate ...')

    firing_rate_map = pd.DataFrame(columns=['trial_number', 'bin_count', 'b_spike_number', 'nb_spike_number','dwell_time_ms'])

    bin_size_cm,number_of_bins = get_bin_size(spatial_data)
    number_of_trials = spatial_data.trial_number.max() # total number of trials

    cluster_index = 5
    #for cluster in range(len(spike_data)):
    #cluster_index = spike_data.cluster_id.values[cluster] - 1

    trials_b = np.array(spike_data.loc[cluster_index].beaconed_trial_number)
    locations_b = np.array(spike_data.loc[cluster_index].beaconed_position_cm)
    trials_nb = np.array(spike_data.loc[cluster_index].nonbeaconed_trial_number)
    locations_nb = np.array(spike_data.loc[cluster_index].nonbeaconed_position_cm)

    for t in range(1,int(number_of_trials)):
        try:
            trial_indices=np.where(trials_b == t)[0]
            trial_locations = np.take(locations_b, trial_indices)
            for loc in range(int(number_of_bins)):
                spikes_in_bin = trial_locations[np.where(np.logical_and(trial_locations > (5*loc), trial_locations < (5*(loc+1))))]
                firing_rate_map = firing_rate_map.append( {"trial_number": int(t), "bin_count": int(loc), "b_spike_number":  len(spikes_in_bin), "nb_spike_number":  0,}, ignore_index=True)
        except ValueError:
            try:
                trial_indices=np.where(trials_nb == t)[0]
                trial_locations = np.take(locations_nb, trial_indices)

                for loc in range(int(number_of_bins)):
                    spikes_in_bin = trial_locations[np.where(np.logical_and(trial_locations > (5*loc), trial_locations < (5*(loc+1))))]
                    firing_rate_map = firing_rate_map.append({ "trial_number": int(t), "bin_count": int(loc), "b_spike_number":  0, "nb_spike_number":  len(spikes_in_bin),}, ignore_index=True)
            except ValueError:
                for loc in range(int(number_of_bins)):
                     firing_rate_map = firing_rate_map.append({"trial_number": int(t),"bin_count": int(loc),"b_spike_number":  0, "nb_spike_number":  0,}, ignore_index=True)

    avg_spike_per_bin_b,avg_spike_per_bin_nb = average_spikes_over_trials(firing_rate_map,number_of_bins)

    spike_data.loc[cluster_index].beaconed_avg_spike_per_bin = list(avg_spike_per_bin_b)
    spike_data.loc[cluster_index].nonbeaconed_avg_spike_per_bin = list(avg_spike_per_bin_nb)
    return spike_data


def make_firing_field_maps(spike_data, spatial_data):
    spike_data = calculate_firing_rate(spike_data, spatial_data)
    return spike_data
