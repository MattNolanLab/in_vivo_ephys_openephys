import numpy as np


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
            spike_data.firing_times[cluster] = spike_data.firing_times[cluster][firing_times_first_half].copy()
            spike_data.position_x[cluster] = np.array(spike_data.position_x[cluster])[firing_times_first_half].copy()
            spike_data.position_y[cluster] = np.array(spike_data.position_y[cluster])[firing_times_first_half].copy()
            spike_data.position_x_pixels[cluster] = np.array(spike_data.position_x_pixels[cluster])[firing_times_first_half].copy()
            spike_data.position_y_pixels[cluster] = np.array(spike_data.position_y_pixels[cluster])[firing_times_first_half].copy()
            spike_data.hd[cluster] = np.array(spike_data.hd[cluster])[firing_times_first_half].copy()
        spike_data_half = spike_data[['firing_times', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd']].copy()

    if half == 'second_half':
        second_half_synced_data_indices = synced_spatial_data.synced_time >= end_of_first_half_seconds
        synced_spatial_data_half = synced_spatial_data[second_half_synced_data_indices]
        for cluster in range(len(spike_data)):
            cluster = spike_data.cluster_id.values[cluster] - 1
            firing_times_second_half = spike_data.firing_times[cluster] >= end_of_first_half_ephys_sampling_points
            spike_data.firing_times[cluster] = spike_data.firing_times[cluster][firing_times_second_half].copy()
            spike_data.position_x[cluster] = np.array(spike_data.position_x[cluster])[firing_times_second_half].copy()
            spike_data.position_y[cluster] = np.array(spike_data.position_y[cluster])[firing_times_second_half].copy()
            spike_data.position_x_pixels[cluster] = np.array(spike_data.position_x_pixels[cluster])[firing_times_second_half].copy()
            spike_data.position_y_pixels[cluster] = np.array(spike_data.position_y_pixels[cluster])[firing_times_second_half].copy()
            spike_data.hd[cluster] = np.array(spike_data.hd[cluster])[firing_times_second_half].copy()
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