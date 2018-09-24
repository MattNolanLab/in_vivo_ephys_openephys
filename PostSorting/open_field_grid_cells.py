import numpy as np
import pandas as pd
import array_utility
import matplotlib.pylab as plt


def calculate_grid_score():
    pass


# shifts array by x and y
def get_shifted_map(firing_rate_map, x, y):
    shifted_map = array_utility.shift_2d(firing_rate_map, x, 0)
    shifted_map = array_utility.shift_2d(shifted_map, y, 1)
    return shifted_map


# remove from both where either of them is not a number (nan)
def remove_zeros(firing_rate_map, shifted_map):
    shifted_map = shifted_map.flatten()
    firing_rate_map = firing_rate_map.flatten()
    shifted_map_tmp = np.take(shifted_map, np.where(firing_rate_map != 0))
    firing_rate_map_tmp = np.take(firing_rate_map, np.where(shifted_map != 0))
    shifted_map = np.take(shifted_map_tmp, np.where(shifted_map_tmp[0] != 0))
    firing_rate_map = np.take(firing_rate_map_tmp, np.where(firing_rate_map_tmp[0] != 0))
    return firing_rate_map.flatten(), shifted_map.flatten()



'''
The array is shifted along the x and y axes into every possible position where it overlaps with itself starting from
the position where the shifted array's bottom right element overlaps with the top left of the map. Correlation is
calculated for all positions and returned as a correlation_vector. TThe correlation vector is 2x * 2y.
'''


def get_rate_map_autocorrelogram(firing_rate_map):
    length_y = firing_rate_map.shape[0] -1
    length_x = firing_rate_map.shape[1] - 1
    correlation_vector = np.empty((length_x * 2 + 1, length_x * 2 + 1)) * 0
    for shift_x in range(-length_x, length_x + 1):
        for shift_y in range(-length_y, length_y + 1):
            # shift map by x and y and remove extra bits
            shifted_map = get_shifted_map(firing_rate_map, shift_x, -shift_y)
            firing_rate_map_to_correlate, shifted_map = remove_zeros(firing_rate_map, shifted_map)

            correlation_y = shift_x + length_x
            correlation_x = shift_y + length_y

            if len(shifted_map) > 20:
                # np.corrcoef(x,y)[0][1] gives the same result for 1d vectors as matlab's corr(x,y) (Pearson)
                # https://stackoverflow.com/questions/16698811/what-is-the-difference-between-matlab-octave-corr-and-python-numpy-correlate
                correlation_vector[correlation_x, correlation_y] = np.corrcoef(firing_rate_map_to_correlate, shifted_map)[0][1]
            else:
                correlation_vector[correlation_x, correlation_y] = np.nan
    return correlation_vector


def process_grid_data(spatial_firing):
    rate_map_correlograms = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        firing_rate_map = spatial_firing.firing_maps[cluster]
        rate_map_correlogram = get_rate_map_autocorrelogram(firing_rate_map)
        rate_map_correlograms.append(rate_map_correlogram)
    spatial_firing['rate_map_autocorrelogram'] = rate_map_correlograms
    return spatial_firing


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # get_correlation_vector(np.array([[1, 1, 1, 1], [2, 2, 2, 9], [3, 3, 3, 3], [4, 4, 4, 4]]))
    # firing_rate_map_matlab = np.genfromtxt('C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of/matlab_rate_map.csv', delimiter=',')
    # get_correlation_vector(firing_rate_map_matlab)

    spatial_firing = pd.read_pickle("C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of/spatial_firing_for_grid.pkl")
    process_grid_data(spatial_firing)


if __name__ == '__main__':
    main()


