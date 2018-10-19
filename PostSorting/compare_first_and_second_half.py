import numpy as np


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
    if number_of_firing_fields > 0:
        firing_field_spike_times = spike_data.spike_times_in_fields[cluster]
        firing_field_times_session = spike_data.times_in_session_fields[cluster]
        for field_id, field in enumerate(firing_field_spike_times):
            if half == 'first':
                firing_times_field = np.take(field, np.where(field < end_of_first_half_ephys_sampling_points))
                mask_firing_times_in_field = np.in1d(spike_data.firing_times[cluster], firing_times_field)
                number_of_spikes_field = mask_firing_times_in_field.sum()
                hd_field_cluster = spike_data.hd[cluster][mask_firing_times_in_field]

            else:
                firing_times_field = np.take(field, np.where(field >= end_of_first_half_ephys_sampling_points))
                mask_firing_times_in_field = np.in1d(spike_data.firing_times[cluster], firing_times_field)
                number_of_spikes_field = mask_firing_times_in_field.sum()
                hd_field_cluster = spike_data.hd[cluster][mask_firing_times_in_field]

        for field_id, field in enumerate(firing_field_times_session):
            if half == 'first':
                times_field = np.take(field, np.where(field < end_of_first_half_seconds))
                mask_times_in_field = np.in1d(synced_spatial_data.synced_time, times_field)
                amount_of_time_spent_in_field = mask_times_in_field.sum()
                hd_field = synced_spatial_data.hd[mask_times_in_field]


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
        spike_data_half = spike_data[['firing_times', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd']].copy()

    if half == 'second_half':
        second_half_synced_data_indices = synced_spatial_data.synced_time >= end_of_first_half_seconds
        synced_spatial_data_half = synced_spatial_data[second_half_synced_data_indices]
        for cluster in range(len(spike_data)):
            cluster = spike_data.cluster_id.values[cluster] - 1
            firing_times_second_half = spike_data.firing_times[cluster] >= end_of_first_half_ephys_sampling_points
            spike_data = get_data_from_data_frame_for_cluster(spike_data, cluster, firing_times_second_half)
            spike_data = get_data_from_data_frames_fields(spike_data, cluster, end_of_first_half_seconds, end_of_first_half_ephys_sampling_points, half='second')
        spike_data_half = spike_data[['firing_times', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd']].copy()
    return spike_data_half, synced_spatial_data_half


def analyse_first_and_second_halves(prm, synced_spatial_data, spike_data_in):
    prm.set_output_path(prm.get_filepath())
    print('---------------------------------------------------------------------------')
    print('I will get data from the first half of the recording.')
    prm.set_output_path(prm.get_filepath() + '/first_half')
    spike_data_first, synced_spatial_data_first = get_half_of_the_data(spike_data_in, synced_spatial_data, half='first_half')
    print('---------------------------------------------------------------------------')
    print('I will get data from the second half of the recording.')
    spike_data_second, synced_spatial_data_second = get_half_of_the_data(spike_data_in, synced_spatial_data, half='second_half')
    prm.set_output_path(prm.get_filepath() + '/second_half')