import numpy as np


def find_firing_location_indices2(spike_data, spatial_data):
    for cluster in range(len(spike_data)):
        location_indices = np.zeros([len(spike_data.firing_times_seconds[cluster])])
        for firing_event in range(len(spike_data.firing_times_seconds[cluster])):
            spatial_index = np.argmin(np.abs(spatial_data.synced_time - spike_data.firing_times_seconds[cluster][firing_event]))
            # np.abs(spatial_data.synced_time - spike_data.firing_times_seconds[cluster][firing_event]).argmin()
            location_indices[firing_event] = spatial_index
        spike_data['spatial_index'][0] = list(location_indices)


def find_firing_location_indices(spike_data, spatial_data):
    for cluster in range(len(spike_data)):
        bonsai_indices_cluster = spike_data.bonsai_indices[cluster]
        bonsai_indices_cluster_round = bonsai_indices_cluster.round(0)
        spike_data = spike_data.append({
            "position_x": spatial_data.position_x[bonsai_indices_cluster_round], # this is still bad - maybe when I removed the beginning of the synced array it didnt change indices
            "position_y":  spatial_data.position_y[bonsai_indices_cluster_round],
            "hd": spatial_data.hd[bonsai_indices_cluster_round]
        }, ignore_index=True)
    return spike_data



def add_firing_locations(spike_data, spatial_data):
    # convert firing times to seconds
    avg_sampling_rate_bonsai = float(1 / spatial_data['synced_time'].diff().mean())
    avg_sampling_rate_open_ephys = 30000
    sampling_rate_rate = avg_sampling_rate_open_ephys/avg_sampling_rate_bonsai
    spike_data['bonsai_indices'] = spike_data.firing_times/sampling_rate_rate
    spike_data = find_firing_location_indices(spike_data, spatial_data)

    # find closest times in spatial data
    # get the corresponding xy values

    # spike_data['location_at_firing_x']
    print('done')



def add_HD_firing(spike_data, spatial_data):
    pass


def process_spatial_firing(spike_data, spatial_data):
    add_firing_locations(spike_data, spatial_data)
