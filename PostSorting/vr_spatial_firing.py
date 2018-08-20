import pandas as pd
import PostSorting.parameters
import numpy as np

prm = PostSorting.parameters.Parameters()


def find_firing_location_indices(spike_data, spatial_data):
    print('I am extracting firing locations...')
    #for cluster in range(len(spike_data)):
    cluster = 5
    #cluster_index = spike_data.cluster_id.values[cluster] - 1
    cluster_firing_indices = spike_data.firing_times[cluster]
    spike_data = spike_data.append({
        "position_cm": list(spatial_data.position_cm[cluster_firing_indices]),
        "trial_number": list(spatial_data.trial_number[cluster_firing_indices]),
        "trial_type":  list(spatial_data.trial_type[cluster_firing_indices])
    }, ignore_index=True)

    print('Firing locations have been extracted for each cluster')
    return spike_data


def process_spatial_firing(spike_data, spatial_data):

    spatial_firing = find_firing_location_indices(spike_data, spatial_data)

    return spatial_firing




def split_spatial_firing_by_trial_type(spike_data):
    cluster_index=41
    cluster_df = spike_data.iloc[[cluster_index]] # dataframe for that cluster
    trials = np.array(cluster_df.trial_number.tolist())
    locations = np.array(cluster_df.position_cm.tolist())
    trial_type = np.array(cluster_df.trial_type.tolist())

    surplus,beaconed_trial_indices=np.where(trial_type == 0)
    surplus,nonbeaconed_trial_indices=np.where(trial_type == 1)
    beaconed_locations = np.take(locations, beaconed_trial_indices)
    nonbeaconed_locations = np.take(locations, nonbeaconed_trial_indices)
    beaconed_trials = np.take(trials, beaconed_trial_indices)
    nonbeaconed_trials = np.take(trials, nonbeaconed_trial_indices)

    spike_data = spike_data.append({
        "beaconed_location": beaconed_locations,
        "beaconed_trial_number": beaconed_trials,
        "nonbeaconed_location": nonbeaconed_locations,
        "nonbeaconed_trial_number": nonbeaconed_trials,
    }, ignore_index=True)

    return spike_data
