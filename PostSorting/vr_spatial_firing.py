import pandas as pd
import PostSorting.parameters

prm = PostSorting.parameters.Parameters()


def find_firing_location_indices(spike_data, spatial_data):
    print('I am extracting firing locations...')
    #for cluster in range(len(spike_data)):
    cluster = 0
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
