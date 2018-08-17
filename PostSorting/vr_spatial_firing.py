import pandas as pd
import PostSorting.parameters

prm = PostSorting.parameters.Parameters()


def find_firing_location_indices(spike_data, spatial_data):
    print('I am extracting firing locations...')
    spatial_firing = pd.DataFrame(columns=['position_cm', 'trial_number', 'trial_type', 'dwell_time_ms'])
    #for cluster in range(len(spike_data)):
    cluster = 5
    cluster_firing_indices = spike_data.firing_times[cluster]
    spatial_firing = spatial_firing.append({
        "position_cm": list(spatial_data.position_cm[cluster_firing_indices]),
        "trial_number": list(spatial_data.trial_number[cluster_firing_indices]),
        "trial_type":  list(spatial_data.trial_type[cluster_firing_indices]),
        "dwell_time_ms":  list(spatial_data.dwell_time_ms[cluster_firing_indices]),
    }, ignore_index=True)
    spike_data['position_cm'] = spatial_firing.position_cm
    spike_data['trial_number'] = spatial_firing.trial_number
    spike_data['trial_type'] = spatial_firing.trial_type
    spike_data['dwell_time_ms'] = spatial_firing.dwell_time_ms
    print('Firing locations have been extracted for each cluster')
    return spike_data


def process_spatial_firing(spike_data, spatial_data):

    spatial_firing = find_firing_location_indices(spike_data, spatial_data)

    return spatial_firing