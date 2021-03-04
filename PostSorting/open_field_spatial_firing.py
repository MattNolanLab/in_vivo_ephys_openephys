import pandas as pd
import setting

def calculate_corresponding_indices(spike_data, spatial_data, Fs=setting.sampling_rate):
    # find the corresponding bonsai data index of each spike
    # multiple spikes can belong to the same bonsai indice
    avg_sampling_rate_bonsai = float(1 / spatial_data['synced_time'].diff().mean())
    avg_sampling_rate_open_ephys = Fs  # Hz
    sampling_rate_rate = avg_sampling_rate_open_ephys/avg_sampling_rate_bonsai
    spike_data['bonsai_indices'] = spike_data.firing_times/sampling_rate_rate
    
    return spike_data

def remove_excess_firing(spike_data, spatial_data,Fs=setting.sampling_rate):
    # remove spike time that are beyond the bonsai data (e.g. when bonsai is stopped before open-ephys)
    bonsai_last_time = spatial_data.synced_time.values[-1] - 1 #make sure we cover the last frame
    for i in range(len(spike_data)):
        #firing_times is in samples, while bonsai time is in second
        spike_data.loc[i,'firing_times'] = spike_data.firing_times[i][spike_data.firing_times[i]< bonsai_last_time*Fs]

    return spike_data

def find_firing_location_indices(spike_data, spatial_data):
    
    spike_data = remove_excess_firing(spike_data, spatial_data)

    spike_data = calculate_corresponding_indices(spike_data, spatial_data)

    spatial_firing = pd.DataFrame(columns=['position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd', 'speed'])
    for cluster in range(len(spike_data)):
        bonsai_indices_cluster = spike_data.bonsai_indices[cluster]
        bonsai_indices_cluster_round = bonsai_indices_cluster.round(0)
        
        spatial_firing = spatial_firing.append({
            "position_x": list(spatial_data.position_x[bonsai_indices_cluster_round]),
            "position_x_pixels": list(spatial_data.position_x_pixels[bonsai_indices_cluster_round]),
            "position_y":  list(spatial_data.position_y[bonsai_indices_cluster_round]),
            "position_y_pixels":  list(spatial_data.position_y_pixels[bonsai_indices_cluster_round]),
            "hd": list(spatial_data.hd[bonsai_indices_cluster_round]),
            "speed": list(spatial_data.speed[bonsai_indices_cluster_round])
        }, ignore_index=True)
    spike_data['position_x'] = spatial_firing.position_x.values
    spike_data['position_x_pixels'] = spatial_firing.position_x_pixels.values
    spike_data['position_y'] = spatial_firing.position_y.values
    spike_data['position_y_pixels'] = spatial_firing.position_y_pixels.values
    spike_data['hd'] = spatial_firing.hd.values
    spike_data['speed'] = spatial_firing.speed.values
    return spike_data


def add_firing_locations(spike_data, spatial_data):
    spike_data = find_firing_location_indices(spike_data, spatial_data)
    spike_data = spike_data.drop(['bonsai_indices'], axis=1)
    return spike_data


def process_spatial_firing(spike_data, spatial_data):
    if 'position_x' in spike_data:
        return spike_data
    spatial_spike_data = add_firing_locations(spike_data, spatial_data)
    return spatial_spike_data
