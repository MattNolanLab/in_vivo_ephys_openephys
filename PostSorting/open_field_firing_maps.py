import matplotlib.pylab as plt
import pandas as pd
from numba import jit
import numpy as np
import math


def get_dwell(spatial_data, prm):
    min_dwell_distance_cm = 5  # from point to determine min dwell time
    min_dwell_distance_pixels = min_dwell_distance_cm / 100 * prm.get_pixel_ratio()

    dt_position_ms = spatial_data.synced_time.diff().mean()*1000  # average sampling interval in position data (ms)
    min_dwell_time_ms = 3 * dt_position_ms  # this is about 100 ms
    min_dwell = round(min_dwell_time_ms/dt_position_ms)
    return min_dwell, min_dwell_distance_pixels


def get_bin_size(prm):
    bin_size_cm = 2.5
    bin_size_pixels = bin_size_cm / 100 * prm.get_pixel_ratio()
    return bin_size_pixels


def get_number_of_bins(spatial_data, prm):
    bin_size_pixels = get_bin_size(prm)
    length_of_arena_x = spatial_data.position_x_pixels.max()
    length_of_arena_y = spatial_data.position_y_pixels.max()
    number_of_bins_x = math.ceil(length_of_arena_x / bin_size_pixels)
    number_of_bins_y = math.ceil(length_of_arena_y / bin_size_pixels)
    return number_of_bins_x, number_of_bins_y


@jit
def gaussian_kernel(kernx):
    kerny = np.exp(np.power(kernx, 2)/2 * (-1))
    return kerny


def calculate_firing_rate_for_cluster(prm, positions_x, positions_y, cluster_firings, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms):
    smooth = 5 / 100 * prm.get_pixel_ratio()
    spike_positions_x = cluster_firings.position_x.values
    spike_positions_y = cluster_firings.position_y.values
    firing_rate_map = np.zeros((number_of_bins_x, number_of_bins_y))
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)
            spike_distances = np.sqrt(np.power(px - spike_positions_x, 2) + np.power(py - spike_positions_y, 2))
            spike_distances = spike_distances[~np.isnan(spike_distances)]
            occupancy_distances = np.sqrt(np.power((px - positions_x), 2) + np.power((py - positions_y), 2))
            occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

            if bin_occupancy >= min_dwell:
                firing_rate_map[x, y] = sum(gaussian_kernel(spike_distances/smooth)) / (sum(gaussian_kernel(occupancy_distances/smooth)) * (dt_position_ms/1000))

            else:
                firing_rate_map[x, y] = 0
    return firing_rate_map


def get_spike_heatmap(spatial_data, firing_data_spatial, prm):
    spatial_firing_maps = pd.DataFrame(columns=['firing_map'])
    dt_position_ms = spatial_data.synced_time.diff().mean()*1000
    min_dwell, min_dwell_distance_pixels = get_dwell(spatial_data, prm)
    smooth = 5 / 100 * prm.get_pixel_ratio()
    bin_size_pixels = get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data, prm)

    for cluster in range(len(firing_data_spatial)):
        cluster_firings = pd.DataFrame({'position_x': firing_data_spatial.position_x_pixels[cluster], 'position_y': firing_data_spatial.position_y_pixels[cluster]})
        firing_rate_map = calculate_firing_rate_for_cluster(prm, spatial_data.position_x_pixels.values, spatial_data.position_y_pixels.values, cluster_firings, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms)
        # plt.imshow(firing_rate_map, cmap='jet', interpolation='nearest')
        spatial_firing_maps = spatial_firing_maps.append({
            "firing_map": np.rot90(firing_rate_map)
        }, ignore_index=True)
    firing_data_spatial['firing_maps'] = spatial_firing_maps.firing_map
    return firing_data_spatial


def get_position_heatmap(spatial_data, prm):
    min_dwell, min_dwell_distance_cm = get_dwell(spatial_data, prm)
    bin_size_cm = get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data, prm)

    position_heat_map = np.zeros((number_of_bins_x, number_of_bins_y))

    # find value for each bin for heatmap
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_cm + (bin_size_cm / 2)
            py = y * bin_size_cm + (bin_size_cm / 2)

            occupancy_distances = np.sqrt(np.power((px - spatial_data.position_x_pixels.values), 2) + np.power((py - spatial_data.position_y_pixels.values), 2))
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_cm)[0])

            if bin_occupancy >= min_dwell:
                position_heat_map[x, y] = bin_occupancy
            else:
                position_heat_map[x, y] = None
    # plt.imshow(position_heat_map, cmap='jet', interpolation='nearest')
    position_heat_map = np.rot90(position_heat_map)
    return position_heat_map


def make_firing_field_maps(spatial_data, firing_data_spatial, prm):
    position_heat_map = get_position_heatmap(spatial_data, prm)
    firing_data_spatial = get_spike_heatmap(spatial_data, firing_data_spatial, prm)

    return position_heat_map, firing_data_spatial