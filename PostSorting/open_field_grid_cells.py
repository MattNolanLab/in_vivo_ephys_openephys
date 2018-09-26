import numpy as np
import pandas as pd
import array_utility
from skimage import measure
from scipy import misc
from scipy.ndimage import rotate
import matplotlib.pylab as plt



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


# make autocorr map binary based on threshold
def threshold_autocorrelation_map(autocorrelation_map):
    autocorrelation_map[autocorrelation_map > 0.2] = 1
    autocorrelation_map[autocorrelation_map <= 0.2] = 0
    return autocorrelation_map


# find peaks of autocorrelogram
def find_autocorrelogram_peaks(autocorrelation_map):
    autocorrelation_map_thresholded = threshold_autocorrelation_map(autocorrelation_map)
    autocorr_map_labels = measure.label(autocorrelation_map_thresholded)  # each field is labelled with a single digit
    field_properties = measure.regionprops(autocorr_map_labels)
    return field_properties


# calculate distances between the middle of the rate map autocorrelogram and the field centres
def find_field_distances_from_mid_point(autocorr_map, field_properties):
    distances = []
    mid_point_coord_x = np.ceil(autocorr_map.shape[0] / 2)
    mid_point_coord_y = np.ceil(autocorr_map.shape[1] / 2)

    for field in range(len(field_properties)):
        field_central_x = field_properties[field].centroid[0]
        field_central_y = field_properties[field].centroid[1]
        distance = np.sqrt(np.square(field_central_x - mid_point_coord_x) + np.square(field_central_y - mid_point_coord_y))
        distances.append(distance)
    return distances


'''
Grid spacing/wavelength:
Defined by Hafting, Fyhn, Molden, Moser, Moser (2005) as the distance from the central autocorrelogram peak to the
vertices of the inner hexagon in the autocorrelogram (the median of the six distances). This should be in cm.
'''


def calculate_grid_spacing(field_distances, bin_size):
    grid_spacing = np.median(field_distances) * bin_size
    return grid_spacing


'''
Defined by Wills, Barry, Cacucci (2012) as the square root of the area of the central peak of the autocorrelogram
divided by pi. (This should be in cm2.)
'''


def calculate_field_size(field_properties, field_distances, bin_size):
    central_field_index = np.argmin(field_distances)
    field_size_pixels = field_properties[central_field_index].area  # number of pixels in central field
    field_size = np.sqrt(field_size_pixels * np.squeeze(bin_size)) / np.pi
    return field_size


# https://stackoverflow.com/questions/481144/equation-for-testing-if-a-point-is-inside-a-circle
def in_circle(center_x, center_y, radius, x, y):
    square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    return square_dist <= radius ** 2


#  replace values not in the grid ring with nan
def remove_inside_and_outside_of_grid_ring(autocorr_map, field_properties, field_distances):
    ring_distances = get_ring_distances(field_distances)
    inner_radius = (np.mean(ring_distances) * 0.5) / 2
    outer_radius = (np.mean(ring_distances) * 2.5) / 2
    center_x = field_properties[np.argmin(field_distances)].centroid[0]
    center_y = field_properties[np.argmin(field_distances)].centroid[1]
    for row in range(autocorr_map.shape[0]):
        for column in range(autocorr_map.shape[1]):
            in_ring = in_circle(center_x, center_y, outer_radius, row, column)
            in_middle = in_circle(center_x, center_y, inner_radius, row, column)
            if not in_ring or in_middle:
                autocorr_map[row, column] = np.nan
    return autocorr_map


def calculate_grid_score(autocorr_map, field_properties, field_distances):
    grid_score = []
    correlation_coefficients = []
    for angle in range(30, 180, 30):
        autocorr_map_to_rotate = np.nan_to_num(autocorr_map)
        rotated_map = rotate(autocorr_map_to_rotate, angle, reshape=False)  # todo fix this
        autocorr_map_ring = remove_inside_and_outside_of_grid_ring(autocorr_map, field_properties, field_distances)
        rotated_map_ring = remove_inside_and_outside_of_grid_ring(rotated_map, field_properties, field_distances)
        pearson_coeff = np.corrcoef(autocorr_map_ring, rotated_map_ring)[0][1]  # todo this has to be the same as corrcoef(amap,amap2,'rows','pairwise');  r = R(1,2);
        correlation_coefficients.append(pearson_coeff)
    grid_score = min(correlation_coefficients[i] for i in [1, 3]) - max(correlation_coefficients[i] for i in [0, 2, 4])

    return grid_score


def get_ring_distances(field_distances_from_mid_point):
    field_distances_from_mid_point = np.array(field_distances_from_mid_point)[~np.isnan(field_distances_from_mid_point)]
    ring_distances = np.sort(field_distances_from_mid_point)[1:7]
    return ring_distances


def calculate_grid_metrics(autocorr_map, field_properties):
    bin_size = 2.5  # cm
    field_distances_from_mid_point = find_field_distances_from_mid_point(autocorr_map, field_properties)
    # the field with the shortest distance is the middle and the next 6 closest are the middle 6
    ring_distances = get_ring_distances(field_distances_from_mid_point)
    grid_spacing = calculate_grid_spacing(ring_distances, bin_size)
    field_size = calculate_field_size(field_properties, field_distances_from_mid_point, bin_size)
    grid_score = calculate_grid_score(autocorr_map, field_properties, field_distances_from_mid_point)
    return grid_spacing, field_size, grid_score


def process_grid_data(spatial_firing):
    rate_map_correlograms = []
    grid_spacings = []
    field_sizes = []
    grid_scores = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        firing_rate_map = spatial_firing.firing_maps[cluster]
        rate_map_correlogram = get_rate_map_autocorrelogram(firing_rate_map)
        rate_map_correlograms.append(rate_map_correlogram)
        field_properties = find_autocorrelogram_peaks(rate_map_correlogram)
        if len(field_properties) > 7:
            grid_spacing, field_size, grid_score = calculate_grid_metrics(rate_map_correlogram, field_properties)
            grid_spacings.append(grid_spacing)
            field_sizes.append(field_size)
            grid_scores.append(grid_score)
        else:
            print('Not enough fields to calculate grid metrics.')
            rate_map_correlograms.append(np.nan)
            grid_spacings.append(np.nan)
            field_sizes.append(np.nan)
            grid_scores.append(np.nan)
    spatial_firing['rate_map_autocorrelogram'] = rate_map_correlograms
    spatial_firing['grid_spacing'] = grid_spacings
    spatial_firing['field_size'] = field_sizes
    spatial_firing['grid_score'] = grid_scores
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


