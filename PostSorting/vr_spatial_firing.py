import pandas as pd
import PostSorting.parameters
import numpy as np

prm = PostSorting.parameters.Parameters()


def add_columns_to_dataframe(spike_data):
    spike_data["position_cm"] = ""
    spike_data["trial_number"] = ""
    spike_data["trial_type"] = ""
    spike_data["beaconed_position_cm"] = ""
    spike_data["beaconed_trial_number"] = ""
    spike_data["nonbeaconed_position_cm"] = ""
    spike_data["nonbeaconed_trial_number"] = ""
    spike_data["avg_spike_per_bin_b"] = ""
    spike_data["avg_spike_per_bin_nb"] = ""
    return spike_data


def find_firing_location_indices(spike_data, spatial_data):
    print('I am extracting firing locations...')
    cluster = 5
    #for cluster in range(len(spike_data)):
    #cluster_index = spike_data.cluster_id.values[cluster] - 1
    cluster_firing_indices = spike_data.firing_times[cluster]

    spike_data.loc[cluster].position_cm = list(spatial_data.position_cm[cluster_firing_indices])
    spike_data.loc[cluster].trial_number = list(spatial_data.trial_number[cluster_firing_indices])
    spike_data.loc[cluster].trial_type = list(spatial_data.trial_type[cluster_firing_indices])
    print('Firing locations have been extracted for each cluster')
    return spike_data


def split_spatial_firing_by_trial_type(spike_data):
    print('Splitting firing locations by trial type...')
    cluster_index=5
    cluster_df = spike_data.iloc[[cluster_index]] # dataframe for that cluster
    trials = np.array(cluster_df.trial_number.tolist())
    locations = np.array(cluster_df.position_cm.tolist())
    trial_type = np.array(cluster_df.trial_type.tolist())
    surplus,beaconed_trial_indices=np.where(trial_type == 0)#find indices where is beaconed trial
    surplus,nonbeaconed_trial_indices=np.where(trial_type == 1)#find indices where is nonbeaconed trial
    beaconed_locations = np.take(locations, beaconed_trial_indices) #split location and trial number
    nonbeaconed_locations = np.take(locations, nonbeaconed_trial_indices)
    beaconed_trials = np.take(trials, beaconed_trial_indices)
    nonbeaconed_trials = np.take(trials, nonbeaconed_trial_indices)

    spike_data.loc[cluster_index].beaconed_position_cm = list(beaconed_locations)
    spike_data.loc[cluster_index].beaconed_trial_number = list(beaconed_trials)
    spike_data.loc[cluster_index].nonbeaconed_position_cm = list(nonbeaconed_locations)
    spike_data.loc[cluster_index].nonbeaconed_trial_number = list(nonbeaconed_trials)
    print('Firing locations have been split by trial type')
    return spike_data


def process_spatial_firing(spike_data, spatial_data):
    spatial_firing = add_columns_to_dataframe(spike_data)
    spatial_firing = find_firing_location_indices(spatial_firing, spatial_data)
    spatial_firing = split_spatial_firing_by_trial_type(spatial_firing)
    return spatial_firing
