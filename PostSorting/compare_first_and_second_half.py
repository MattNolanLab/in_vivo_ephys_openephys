import numpy as np
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots


def get_data_from_data_frame_for_cluster(spike_data, cluster, indices):
    spike_data.firing_times[cluster] = spike_data.firing_times[cluster][indices].copy()
    spike_data.position_x[cluster] = np.array(spike_data.position_x[cluster])[indices].copy()
    spike_data.position_y[cluster] = np.array(spike_data.position_y[cluster])[indices].copy()
    spike_data.position_x_pixels[cluster] = np.array(spike_data.position_x_pixels[cluster])[indices].copy()
    spike_data.position_y_pixels[cluster] = np.array(spike_data.position_y_pixels[cluster])[indices].copy()
    spike_data.hd[cluster] = np.array(spike_data.hd[cluster])[indices].copy()
    return spike_data


def get_data_from_data_frames_fields(spike_data, synced_spatial_data, cluster, end_of_first_half_seconds, end_of_first_half_ephys_sampling_points, half='first'):
    number_of_firing_fields = len(spike_data.firing_fields[cluster])
    number_of_spikes_in_fields = []
    number_of_samples_in_fields = []
    hd_in_fields_cluster = []
    hd_in_field_sessions = []
    if number_of_firing_fields > 0:
        firing_field_spike_times = spike_data.spike_times_in_fields[cluster]
        firing_field_times_session = spike_data.times_in_session_fields[cluster]
        for field_id, field in enumerate(firing_field_spike_times):
            if half == 'first':
                firing_times_field = np.take(field, np.where(field < end_of_first_half_ephys_sampling_points))
                mask_firing_times_in_field = np.in1d(spike_data.firing_times[cluster], firing_times_field)

            else:
                firing_times_field = np.take(field, np.where(field >= end_of_first_half_ephys_sampling_points))
                mask_firing_times_in_field = np.in1d(spike_data.firing_times[cluster], firing_times_field)
            number_of_spikes_field = mask_firing_times_in_field.sum()
            hd_field_cluster = spike_data.hd[cluster][mask_firing_times_in_field]
            hd_field_cluster = (np.array(hd_field_cluster) + 180) * np.pi / 180
            hd_fields_cluster_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_field_cluster)
            number_of_spikes_in_fields.append(number_of_spikes_field)
            hd_in_fields_cluster.append(hd_fields_cluster_hist)

        for field_id, field in enumerate(firing_field_times_session):
            if half == 'first':
                times_field = np.take(field, np.where(field < end_of_first_half_seconds))
                mask_times_in_field = np.in1d(synced_spatial_data.synced_time, times_field)
            else:
                times_field = np.take(field, np.where(field >= end_of_first_half_seconds))
                mask_times_in_field = np.in1d(synced_spatial_data.synced_time, times_field)
            amount_of_time_spent_in_field = mask_times_in_field.sum()
            hd_field = synced_spatial_data.hd[mask_times_in_field]
            hd_field = (np.array(hd_field) + 180) * np.pi / 180
            number_of_samples_in_fields.append(amount_of_time_spent_in_field)
            hd_field_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_field)
            hd_in_field_sessions.append(hd_field_hist)

    else:
        number_of_spikes_in_fields.append(None)
        number_of_samples_in_fields.append(None)
        hd_in_fields_cluster.append([None])
        hd_in_field_sessions.append([None])

    spike_data.number_of_spikes_in_fields[cluster] = number_of_firing_fields
    spike_data.time_spent_in_fields_sampling_points[cluster] = number_of_samples_in_fields
    spike_data.firing_fields_hd_cluster[cluster] = hd_in_fields_cluster
    spike_data.firing_fields_hd_session[cluster] = hd_in_field_sessions

    return spike_data


def get_half_of_the_data(spike_data_in, synced_spatial_data_in, half='first_half'):
    spike_data = spike_data_in.copy()
    synced_spatial_data = synced_spatial_data_in.copy()
    synced_spatial_data_half = None
    spike_data_half = None
    end_of_first_half_seconds = (synced_spatial_data.synced_time.max() - synced_spatial_data.synced_time.min()) / 2
    end_of_first_half_ephys_sampling_points = end_of_first_half_seconds * 30000
    if half == 'first_half':
        first_half_synced_data_indices = synced_spatial_data.synced_time < end_of_first_half_seconds
        synced_spatial_data_half = synced_spatial_data[first_half_synced_data_indices].copy()
        for cluster in range(len(spike_data)):
            cluster = spike_data.cluster_id.values[cluster] - 1
            firing_times_first_half = spike_data.firing_times[cluster] < end_of_first_half_ephys_sampling_points
            spike_data = get_data_from_data_frame_for_cluster(spike_data, cluster, firing_times_first_half)
            spike_data = get_data_from_data_frames_fields(spike_data, synced_spatial_data, cluster, end_of_first_half_seconds, end_of_first_half_ephys_sampling_points, half='first')
        spike_data_half = spike_data[['cluster_id', 'firing_times', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd', 'number_of_spikes_in_fields', 'time_spent_in_fields_sampling_points', 'firing_fields_hd_cluster', 'firing_fields_hd_session']].copy()

    if half == 'second_half':
        second_half_synced_data_indices = synced_spatial_data.synced_time >= end_of_first_half_seconds
        synced_spatial_data_half = synced_spatial_data[second_half_synced_data_indices]
        for cluster in range(len(spike_data)):
            cluster = spike_data.cluster_id.values[cluster] - 1
            firing_times_second_half = spike_data.firing_times[cluster] >= end_of_first_half_ephys_sampling_points
            spike_data = get_data_from_data_frame_for_cluster(spike_data, cluster, firing_times_second_half)
            spike_data = get_data_from_data_frames_fields(spike_data, synced_spatial_data, cluster, end_of_first_half_seconds, end_of_first_half_ephys_sampling_points, half='second')
        spike_data_half = spike_data[['cluster_id', 'firing_times', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd', 'number_of_spikes_in_fields', 'time_spent_in_fields_sampling_points', 'firing_fields_hd_cluster', 'firing_fields_hd_session']].copy()
    return spike_data_half, synced_spatial_data_half


def analyse_first_and_second_halves(prm, synced_spatial_data, spike_data_in):
    print('---------------------------------------------------------------------------')
    print('I will get data from the first half of the recording.')
    prm.set_output_path(prm.get_filepath() + '/first_half')
    spike_data_first, synced_spatial_data_first = get_half_of_the_data(spike_data_in, synced_spatial_data, half='first_half')
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spike_data_first, synced_spatial_data_first, prm)
    print('---------------------------------------------------------------------------')
    print('I will get data from the second half of the recording.')
    spike_data_second, synced_spatial_data_second = get_half_of_the_data(spike_data_in, synced_spatial_data, half='second_half')
    prm.set_output_path(prm.get_filepath() + '/second_half')
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spike_data_second, synced_spatial_data_second, prm)