import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import math


def get_position_heatmap2(spatial_data):
    position = spatial_data[['position_x', 'position_y']]

    min_position_x = min(position.position_x)
    max_position_x = max(position.position_x)
    number_of_bins_x = int(math.ceil((max_position_x - min_position_x) / 2))

    min_position_y = min(position.position_y)
    max_position_y = max(position.position_x)
    number_of_bins_y = int(math.ceil((max_position_y - min_position_y) / 2))

    x_cut = pd.cut(position.position_x, np.linspace(min_position_x, max_position_x, number_of_bins_x), right=False)
    y_cut = pd.cut(position.position_y, np.linspace(min_position_y, max_position_y, number_of_bins_y), right=False)
    heatmap = np.nan_to_num(position.groupby([x_cut, y_cut]).count().position_x.values)
    heatmap = np.reshape(heatmap, (number_of_bins_x - 1, number_of_bins_y - 1))
    plt.imshow(heatmap, cmap='jet', interpolation='nearest')
    plt.show()
    return heatmap


def get_spike_heatmaps2(firing_data_spatial):
    spatial_firing_maps = pd.DataFrame(columns=['firing_map'])
    for cluster in range(len(firing_data_spatial)):
        cluster_firings = pd.DataFrame({'position_x': firing_data_spatial.position_x[cluster], 'position_y': firing_data_spatial.position_y[cluster]})
        firing_heatmap = get_position_heatmap(cluster_firings)
        spatial_firing_maps = spatial_firing_maps.append({
            "firing_map": firing_heatmap
        }, ignore_index=True)
    firing_data_spatial['firing_maps'] = spatial_firing_maps.firing_map
    return firing_data_spatial

###########################################################################################################################################


def get_dwell(spatial_data, prm):
    min_dwell_distance_cm = 5  # from point to determine min dwell time
    min_dwell_distance_cm = min_dwell_distance_cm / 100 * prm.get_pixel_ratio()

    dt_position_ms = spatial_data.synced_time.diff().mean()*1000  # average sampling interval in position data (ms)
    min_dwell_time_ms = 3 * dt_position_ms  # this is about 100 ms
    min_dwell = round(min_dwell_time_ms/dt_position_ms)
    return min_dwell, min_dwell_distance_cm


def get_bin_size(prm):
    bin_size_cm = 2.5
    bin_size_cm = bin_size_cm / 100 * prm.get_pixel_ratio()
    return bin_size_cm


def get_number_of_bins(spatial_data, prm):
    bin_size_cm = get_bin_size(prm)
    length_of_arena_x = spatial_data.position_x.max()
    length_of_arena_y = spatial_data.position_y.max()
    number_of_bins_x = math.ceil(length_of_arena_x / bin_size_cm)
    number_of_bins_y = math.ceil(length_of_arena_y / bin_size_cm)
    return number_of_bins_x, number_of_bins_y


def gaussian_kernel(kernx):
    kerny = np.exp(np.power(-kernx, 2)/2)
    return kerny


def get_spike_heatmap(spatial_data, firing_data_spatial, prm):
    dt_position_ms = spatial_data.synced_time.diff().mean()*1000
    min_dwell, min_dwell_distance_cm = get_dwell(spatial_data, prm)
    smooth = 5 / 100 * prm.get_pixel_ratio()
    bin_size_cm = get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data, prm)
    firing_rate_map = np.zeros((number_of_bins_x, number_of_bins_y))
    for cluster in range(len(firing_data_spatial)):
        cluster_firings = pd.DataFrame({'position_x': firing_data_spatial.position_x[cluster], 'position_y': firing_data_spatial.position_y[cluster]})

        for x in range(number_of_bins_x):
            for y in range(number_of_bins_y):
                px = x * bin_size_cm - (bin_size_cm / 2)
                py = y * bin_size_cm - (bin_size_cm / 2)
                spike_distances = np.sqrt(np.power(px - cluster_firings.position_x.values, 2) + np.power(py - cluster_firings.position_y.values, 2))
                occupancy_distances = np.sqrt(np.power((px - spatial_data.position_x.values), 2) + np.power((py - spatial_data.position_y.values), 2))
                bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_cm)[0])

                if bin_occupancy >= min_dwell:
                    firing_rate_map[x, y] = sum(gaussian_kernel(spike_distances/smooth)) / (sum(gaussian_kernel(occupancy_distances/smooth)) * (dt_position_ms/1000))
                else:
                    firing_rate_map[x, y] = None
        plt.imshow(firing_rate_map, cmap='jet', interpolation='nearest')
    return firing_rate_map


def get_position_heatmap(spatial_data, prm):
    min_dwell, min_dwell_distance_cm = get_dwell(spatial_data, prm)
    bin_size_cm = get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data, prm)

    position_heat_map = np.zeros((number_of_bins_x, number_of_bins_y))

    # find value for each bin for heatmap
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_cm - (bin_size_cm / 2)
            py = y * bin_size_cm - (bin_size_cm / 2)

            occupancy_distances = np.sqrt(np.power((px - spatial_data.position_x.values), 2) + np.power((py - spatial_data.position_y.values), 2))
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_cm)[0])

            if bin_occupancy >= min_dwell:
                position_heat_map[x, y] = bin_occupancy
            else:
                position_heat_map[x, y] = None
    # plt.imshow(position_heat_map, cmap='jet', interpolation='nearest')
    return position_heat_map


def make_firing_field_maps(spatial_data, firing_data_spatial, prm):
    position_heat_map = []
    position_heat_map = get_position_heatmap(spatial_data, prm)
    firing_rate_map = get_spike_heatmap(spatial_data, firing_data_spatial, prm)
    # position_heat_map = get_position_heatmap(spatial_data)
    # spike_heat_maps = get_spike_heatmaps(firing_data_spatial)
    # plt.imshow(spike_heat_maps.firing_maps[5] / position_heat_map, cmap='jet', interpolation='lanczos')

    return position_heat_map