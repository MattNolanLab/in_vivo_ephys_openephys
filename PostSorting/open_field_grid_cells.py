import numpy as np
import array_utility


def calculate_grid_score():
    pass


def get_shifted_map(firing_rate_map, x, y):
    shifted_map = array_utility.shift_2d(firing_rate_map, x, 0)
    shifted_map = array_utility.shift_2d(shifted_map, y, 1)

    return shifted_map


def remove_edges_and_nans(firing_rate_map, shifted_map, shift_x, shift_y):
    #shifted_indices = np.where(shifted_map

    return firing_rate_map, shifted_map


def get_correlation_vector(firing_rate_map):
    length_y = firing_rate_map.shape[0] - 1
    length_x = firing_rate_map.shape[1] - 1
    correlation_vector = np.empty((length_x, length_y,)) * np.nan
    for shift_x in range(-length_x, length_x):
        for shift_y in range(-length_y, length_y):
            # shift map by x and y and remove extra bits
            shifted_map = get_shifted_map(firing_rate_map, shift_x, shift_y)

            correlation_y = shift_x + length_x + 1
            correlation_x = shift_y + length_y + 1

            if len(shifted_map) > 20:
                correlation_vector[correlation_x, correlation_y] = np.correlate(firing_rate_map, shifted_map)

    return correlation_vector




def process_grid_data(spatial_firing):
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        firing_rate_map = spatial_firing.firing_maps[cluster]
        correlation_vector = get_correlation_vector(firing_rate_map)


