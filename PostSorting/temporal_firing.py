import numpy as np

# calculate number of spikes and mean firing rate for each cluster and add to spatial firing df
def add_temporal_firing_properties_to_df(spatial_firing, prm):
    total_number_of_spikes_per_cluster = []
    mean_firing_rates = []
    mean_firing_rates_local = []
    for cluster, cluster_id in enumerate(spatial_firing.cluster_id):
        firing_times = np.asarray(spatial_firing[spatial_firing.cluster_id == cluster_id].firing_times)[0]
        total_number_of_spikes = len(firing_times)
        total_length_of_recording = prm.get_total_length_sampling_points()  # this does not include opto

        if prm.stitchpoint is not None:
            total_length_of_recordings = prm.get_total_length_sampling_points()  # this does not include opto
            mean_firing_rate = total_number_of_spikes / total_length_of_recordings

            if prm.paired_order == "first":
                firing_times = firing_times[firing_times > 0]
            elif prm.paired_order == "second":
                firing_times = firing_times[firing_times < prm.stitchpoint]

            total_number_of_spikes_local = len(firing_times)
            mean_firing_rate_local = total_number_of_spikes_local/total_length_of_recording
            mean_firing_rates_local.append(mean_firing_rate_local)
        else:
            mean_firing_rate = total_number_of_spikes / total_length_of_recording  # this does not include opto

        total_number_of_spikes_per_cluster.append(total_number_of_spikes)
        mean_firing_rates.append(mean_firing_rate)

    spatial_firing['number_of_spikes'] = total_number_of_spikes_per_cluster
    spatial_firing['mean_firing_rate'] = mean_firing_rates
    if prm.stitchpoint is not None:
        spatial_firing['mean_firing_rate_local'] = mean_firing_rates_local

    return spatial_firing

def correct_for_stitch(spatial_firing, prm):
    if prm.paired_order is not None:

        for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
            firing_times = np.asarray(spatial_firing[spatial_firing.cluster_id == cluster_id].firing_times)[0]

            if prm.paired_order == "first":
                spatial_firing.firing_times.iloc[cluster_index] = firing_times[firing_times > 0]
            elif prm.paired_order == "second":
                spatial_firing.firing_times.iloc[cluster_index] = firing_times[firing_times < prm.stitchpoint]

    return spatial_firing

