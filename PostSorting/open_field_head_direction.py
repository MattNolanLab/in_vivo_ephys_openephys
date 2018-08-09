import math
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import PostSorting.open_field_firing_maps


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]


def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window for head-direction histogram is too big, HD plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out


def get_hd_histogram(angles):
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    binned_hd, _, _ = plt.hist(angles, theta)
    smooth_hd = get_rolling_sum(binned_hd, window=23)
    #ax = plt.subplot(1, 1, 1, polar=True)
    #ax.grid(True)
    #ax.plot(theta[:-1], smooth_hd)
    return smooth_hd


def process_hd_data(spatial_firing, spatial_data, prm):
    angles_whole_session = (np.array(spatial_data.hd) + 180) * np.pi / 180
    hd_histogram = get_hd_histogram(angles_whole_session)
    hd_histogram /= prm.get_sampling_rate()

    hd_spike_histograms = []
    for cluster in range(len(spatial_firing)):
        angles_spike = (np.array(spatial_firing.hd[cluster]) + 180) * np.pi / 180
        hd_spike_histogram = get_hd_histogram(angles_spike)
        hd_spike_histogram = hd_spike_histogram / hd_histogram
        hd_spike_histograms.append(hd_spike_histogram)

    spatial_firing['hd_spike_histogram'] = hd_spike_histograms

    return hd_histogram, spatial_firing


def get_indices_for_bin(bin, rate_map_indices, spatial_data, prm):
    bin_size_pixels = PostSorting.open_field_firing_maps.get_bin_size(prm)
    bin_x = bin[0]
    bin_x_left_pixels = bin_x * bin_size_pixels
    bin_x_right_pixels = bin_x * (bin_size_pixels + 1)
    bin_y = bin[1]
    bin_y_bottom_pixels = bin_y * bin_size_pixels
    bin_y_top_pixels = bin_y * (bin_size_pixels + 1)

    left_x_border = spatial_data.x > bin_x_left_pixels
    right_x_border = spatial_data.x < bin_x_right_pixels
    bottom_y_border = spatial_data.y > bin_y_bottom_pixels
    top_y_border = spatial_data.y < bin_y_top_pixels

    inside_bin = spatial_data[left_x_border & right_x_border & bottom_y_border & top_y_border]
    return inside_bin


# get head-direction data from bins of field
def get_hd_in_field(rate_map_indices, spatial_data, prm):
    hd_in_field = []
    for bin_in_field in rate_map_indices:
        inside_bin = get_indices_for_bin(bin_in_field, rate_map_indices, spatial_data, prm)
        hd = inside_bin.hd.values
        hd_in_field.append(hd)
    return hd_in_field


# return array of HD in subfield when cell fired for cluster
def get_hd_in_firing_rate_bins_for_cluster(spatial_firing, rate_map_indices, cluster, prm):
    cluster_id = np.arange(len(spatial_firing.firing_times[cluster]))
    spatial_firing_cluster = pd.DataFrame(cluster_id)
    spatial_firing_cluster['x'] = spatial_firing.position_x_pixels[cluster]
    spatial_firing_cluster['y'] = spatial_firing.position_y_pixels[cluster]
    spatial_firing_cluster['hd'] = spatial_firing.hd[cluster]

    hd_in_field = get_hd_in_field(rate_map_indices, spatial_firing_cluster, prm)
    return hd_in_field


def get_hd_in_firing_rate_bins_for_session(spatial_data, rate_map_indices, prm):
    spatial_data_field = pd.DataFrame()
    spatial_data_field['x'] = spatial_data.position_x_pixels
    spatial_data_field['y'] = spatial_data.position_y_pixels
    spatial_data_field['hd'] = spatial_data.hd
    hd_in_field = get_hd_in_field(rate_map_indices, spatial_data_field, prm)
    return hd_in_field


def main():
    array_in = [3, 4, 5, 8, 11, 1, 3, 5, 4]
    window = 3
    get_rolling_sum(array_in, window)

    array_in = [1, 2, 3, 4, 5, 6]
    window = 3
    get_rolling_sum(array_in, window)

    array_in = [3, 4, 5, 8, 11, 1, 3, 5]
    window = 5
    get_rolling_sum(array_in, window)


if __name__ == '__main__':
    main()