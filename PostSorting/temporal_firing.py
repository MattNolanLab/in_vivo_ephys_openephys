import numpy as np
import settings


def add_temporal_firing_properties_to_df(spatial_firing, stitchpoint, paired_order, total_length_sampling_points):
    # calculate number of spikes and mean firing rate for each cluster and add to spatial firing df
    total_number_of_spikes_per_cluster = []
    mean_firing_rates = []
    mean_firing_rates_local = []
    for cluster, cluster_id in enumerate(spatial_firing.cluster_id):
        firing_times = np.asarray(spatial_firing[spatial_firing.cluster_id == cluster_id].firing_times)[0]
        total_number_of_spikes = len(firing_times)
        total_length_of_recording = total_length_sampling_points/settings.sampling_rate  # this does not include opto

        if stitchpoint is not None:
            total_length_of_recordings = total_length_sampling_points / settings.sampling_rate  # this does not include opto
            mean_firing_rate = total_number_of_spikes / total_length_of_recordings

            if paired_order == 1:
                firing_times = firing_times[firing_times < stitchpoint[0]]
            else:
                bigger_than_previous_stitch = stitchpoint[paired_order - 2] < firing_times
                smaller_than_next = firing_times < stitchpoint[paired_order - 1]
                firing_times = firing_times[bigger_than_previous_stitch & smaller_than_next]

            total_number_of_spikes_local = len(firing_times)
            mean_firing_rate_local = total_number_of_spikes_local/total_length_of_recording
            mean_firing_rates_local.append(mean_firing_rate_local)
        else:
            mean_firing_rate = total_number_of_spikes / total_length_of_recording  # this does not include opto

        total_number_of_spikes_per_cluster.append(total_number_of_spikes)
        mean_firing_rates.append(mean_firing_rate)

    spatial_firing['number_of_spikes'] = total_number_of_spikes_per_cluster
    spatial_firing['mean_firing_rate'] = mean_firing_rates
    if stitchpoint is not None:
        spatial_firing['mean_firing_rate_local'] = mean_firing_rates_local

    return spatial_firing


def correct_for_stitch(spatial_firing, paired_order, stitchpoint):
    if paired_order is not None:
        for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
            firing_times = np.asarray(spatial_firing[spatial_firing.cluster_id == cluster_id].firing_times)[0]
            spatial_firing = get_firing_times_for_recording(paired_order, spatial_firing, cluster_index, firing_times, stitchpoint)
    return spatial_firing


def get_firing_times_for_recording(paired_order, spatial_firing, cluster_index, firing_times, stitchpoint):
    if paired_order == 1:
        spatial_firing.firing_times.iloc[cluster_index] = firing_times[firing_times < stitchpoint[0]]
    else:
        bigger_than_previous_stitch = stitchpoint[paired_order - 2] < firing_times
        smaller_than_next = firing_times < stitchpoint[paired_order - 1]
        firing_times = firing_times[bigger_than_previous_stitch & smaller_than_next]

        spatial_firing.firing_times.iloc[cluster_index] = firing_times
    return spatial_firing



