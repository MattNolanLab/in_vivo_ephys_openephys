import PostSorting.parameters
import numpy as np
import pandas as pd

prm = PostSorting.parameters.Parameters()
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


def average_spikes_over_trials(firing_rate_map,number_of_bins, beaconed_trial_no, nonbeaconed_trial_no, probe_trial_no):
    avg_spikes_across_trials_b = np.zeros((len(range(int(number_of_bins)))))
    avg_spikes_across_trials_nb = np.zeros((len(range(int(number_of_bins)))))
    avg_spikes_across_trials_p = np.zeros((len(range(int(number_of_bins)))))
    number_of_trials = firing_rate_map.trial_number.max() # total number of trials
    for loc in range(int(number_of_bins)):
        try:
            spikes_across_trials_b=sum(firing_rate_map.loc[firing_rate_map.bin_count == loc, 'b_spike_number'])/beaconed_trial_no
            avg_spikes_across_trials_b[loc] = spikes_across_trials_b
        except ZeroDivisionError:
            continue
        try:
            spikes_across_trials_nb=sum(firing_rate_map.loc[firing_rate_map.bin_count == loc, 'nb_spike_number'])/nonbeaconed_trial_no
            avg_spikes_across_trials_nb[loc] = spikes_across_trials_nb
            spikes_across_trials_p=sum(firing_rate_map.loc[firing_rate_map.bin_count == loc, 'p_spike_number'])/probe_trial_no
            avg_spikes_across_trials_p[loc] = spikes_across_trials_p
        except ZeroDivisionError:
            continue
    avg_spikes_across_trials_b = PostSorting.vr_spatial_data.get_rolling_sum(np.nan_to_num(avg_spikes_across_trials_b), 10)
    avg_spikes_across_trials_nb = PostSorting.vr_spatial_data.get_rolling_sum(np.nan_to_num(avg_spikes_across_trials_nb), 10)
    avg_spikes_across_trials_p = PostSorting.vr_spatial_data.get_rolling_sum(np.nan_to_num(avg_spikes_across_trials_p), 10)
    return avg_spikes_across_trials_b,avg_spikes_across_trials_nb,avg_spikes_across_trials_p


def calculate_firing_rate(spike_data, spatial_data):
    print('I am calculating the average firing rate ...')
    bin_size_cm,number_of_bins = get_bin_size(spatial_data)
    number_of_trials = spatial_data.trial_number.max() # total number of trials

    for cluster_index in range(len(spike_data)):
        firing_rate_map = pd.DataFrame(columns=['trial_number', 'bin_count', 'b_spike_number', 'nb_spike_number','p_spike_number','dwell_time_ms'])
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        trials_b = np.array(spike_data.at[cluster_index, 'beaconed_trial_number']);locations_b = np.array(spike_data.at[cluster_index, 'beaconed_position_cm'])
        trials_nb = np.array(spike_data.at[cluster_index,'nonbeaconed_trial_number']);locations_nb = np.array(spike_data.at[cluster_index, 'nonbeaconed_position_cm'])
        trials_p = np.array(spike_data.at[cluster_index, 'probe_trial_number']);locations_p = np.array(spike_data.at[cluster_index, 'probe_position_cm'])
        beaconed_trial_no = len(np.unique(trials_b))
        nonbeaconed_trial_no = len(np.unique(trials_nb))
        probe_trial_no = len(np.unique(trials_p))
        for t in range(1,int(number_of_trials)):
            try:
                trial_locations_b = np.take(locations_b, np.where(trials_b == t)[0])
                trial_locations_nb = np.take(locations_nb, np.where(trials_nb == t)[0])
                trial_locations_p = np.take(locations_p, np.where(trials_p == t)[0])
                if len(trial_locations_b) > 1:
                    for loc in range(int(number_of_bins)):
                        spikes_in_bin_b = trial_locations_b[np.where(np.logical_and(trial_locations_b > float(loc), trial_locations_b <= (loc+1)))]
                        spikes_in_bin_nb = trial_locations_nb[np.where(np.logical_and(trial_locations_nb > loc, trial_locations_nb <= (loc+1)))]
                        spikes_in_bin_p = trial_locations_p[np.where(np.logical_and(trial_locations_p > loc, trial_locations_p <= (loc+1)))]
                        firing_rate_map = firing_rate_map.append({"trial_number": int(t), "bin_count": int(loc), "b_spike_number":  len(spikes_in_bin_b), "nb_spike_number":  len(spikes_in_bin_nb), "p_spike_number":  len(spikes_in_bin_p)}, ignore_index=True)
                else:
                    for loc in range(int(number_of_bins)):
                        firing_rate_map = firing_rate_map.append({"trial_number": int(t), "bin_count": int(loc), "b_spike_number": 0, "nb_spike_number": 0,"p_spike_number": 0}, ignore_index=True)

            except IndexError: # if there is no spikes on that trial
                for loc in range(int(number_of_bins)):
                    firing_rate_map = firing_rate_map.append({"trial_number": int(t),"bin_count": int(loc),"b_spike_number":  0, "nb_spike_number":  0, "p_spike_number":  0}, ignore_index=True)

        firing_rate_map['dwell_time'] = spatial_data['binned_time_ms']
        firing_rate_map['b_spike_number'] = np.where(firing_rate_map['b_spike_number'] > 0, firing_rate_map['b_spike_number']/firing_rate_map['dwell_time'], 0)
        firing_rate_map['nb_spike_number'] = np.where(firing_rate_map['nb_spike_number'] > 0, firing_rate_map['nb_spike_number']/firing_rate_map['dwell_time'], 0)
        firing_rate_map['p_spike_number'] = np.where(firing_rate_map['p_spike_number'] > 0, firing_rate_map['p_spike_number']/firing_rate_map['dwell_time'], 0)

        avg_spike_per_bin_b,avg_spike_per_bin_nb,avg_spike_per_bin_p = average_spikes_over_trials(firing_rate_map,number_of_bins, beaconed_trial_no, nonbeaconed_trial_no, probe_trial_no)

        spike_data.at[cluster_index, 'avg_spike_per_bin_b'] = list(avg_spike_per_bin_b)
        spike_data.at[cluster_index, 'avg_spike_per_bin_nb'] = list(avg_spike_per_bin_nb)
        spike_data.at[cluster_index, 'avg_spike_per_bin_p'] = list(avg_spike_per_bin_p)
    return spike_data


def make_firing_field_maps(spike_data, spatial_data):
    spike_data = calculate_firing_rate(spike_data, spatial_data)
    return spike_data
