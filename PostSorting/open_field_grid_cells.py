import numpy as np


def calculate_grid_score():
    pass


def get_shifted_map(firing_rate_map, x, y):
    shifted_map = []
    return shifted_map


def get_correlation_vector(firing_rate_map):
    length_y = firing_rate_map.shape[0]
    length_x = firing_rate_map.shape[1]
    correlation_vector = np.empty((length_x, length_y,)) * np.nan
    for x in range(2 * length_x - 1):
        for y in range(2 * length_y - 1):
            # shift map by x and y and remove extra bits
            shifted_map = get_shifted_map(firing_rate_map, x, y)

            correlation_y = x + length_x + 1
            correlation_x = y + length_y + 1

            if len(shifted_map) > 20:
                correlation_vector[correlation_x, correlation_y] = np.correlate(firing_rate_map, shifted_map)

    return correlation_vector
