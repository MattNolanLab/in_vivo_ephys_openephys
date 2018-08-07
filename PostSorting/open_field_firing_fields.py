import numpy as np
import pandas as pd

import matplotlib.pylab as plt


def find_neighbors(bin_to_test, max_x, max_y):
    x = bin_to_test[0]
    y = bin_to_test[1]

    neighbors = [[x, y+1], [x, y-1], [x+1, y], [x-1, y]]

    if x == max_x:
        neighbors = [[x, y+1], [x, y-1], [x-1, y]]
    if y == max_y:
        neighbors = [[x, y-1], [x+1, y], [x-1, y]]
    if x == max_x and y == max_y:
        neighbors = [[x, y-1], [x-1, y]]
    if x == 0:
        neighbors = [[x, y+1], [x, y-1], [x+1, y]]
    if y == 0:
        neighbors = [[x, y+1], [x+1, y], [x-1, y]]
    if x == 0 and y == 0:
        neighbors = [[x, y+1], [x+1, y]]

    return neighbors


def find_neighborhood(masked_rate_map, rate_map, firing_rate_of_max):
    changed = False
    threshold = firing_rate_of_max * 80 / 100

    firing_field_bins = np.array(np.where(masked_rate_map == True))
    firing_field_bins = firing_field_bins.T

    for bin_to_test in firing_field_bins:
        masked_rate_map[bin_to_test[0], bin_to_test[1]] = 2
        neighbors = find_neighbors(bin_to_test, max_x=(masked_rate_map.shape[0]-1), max_y=(masked_rate_map.shape[1]-1))
        for neighbor in neighbors:
            if masked_rate_map[neighbor[0], neighbor[1]] == 2:
                continue

            firing_rate = rate_map[neighbor[0], neighbor[1]]
            if firing_rate >= threshold:
                masked_rate_map[neighbor[0], neighbor[1]] = True
                changed = True

    return masked_rate_map, changed


def find_current_maxima_indices(rate_map):
    higest_rate_bin = np.unravel_index(rate_map.argmax(), rate_map.shape)
    plt.imshow(rate_map)
    plt.scatter(higest_rate_bin[1], higest_rate_bin[0], marker='o', s=500, color='yellow')

    masked_rate_map = np.full((rate_map.shape[0], rate_map.shape[1]), False)
    masked_rate_map[higest_rate_bin] = True
    changed = True
    tested_bins = []
    while changed:
        masked_rate_map, changed = find_neighborhood(masked_rate_map, rate_map, rate_map[higest_rate_bin])

    field_indices = np.where(masked_rate_map > 0).T

    return field_indices


def analyze_firing_fields(spatial_firing):
    cluster = 5
    rate_map = spatial_firing.firing_maps[cluster]
    field = find_current_maxima_indices(rate_map)

    #  check which other indices belong there





def main():
    firing_rate_maps = np.load('C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of/M5_2018-03-06_15-34-44_of.npy')
    cluster_id = np.arange(len(firing_rate_maps))
    spatial_firing = pd.DataFrame(cluster_id)
    spatial_firing['firing_maps'] = list(firing_rate_maps)
    analyze_firing_fields(spatial_firing)

if __name__ == '__main__':
    main()