import math
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import kuiper

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
    plt.figure()
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    binned_hd, _, _ = plt.hist(angles, theta)
    smooth_hd = get_rolling_sum(binned_hd, window=23)
    plt.close()
    return smooth_hd


# max firing rate at the angle where the firing rate is highest
def get_max_firing_rate(spatial_firing):
    max_firing_rates = []
    preferred_directions = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        hd_hist = spatial_firing.hd_spike_histogram[cluster]
        max_firing_rate = np.max(hd_hist.flatten())
        max_firing_rates.append(max_firing_rate)

        preferred_direction = np.where(hd_hist == max_firing_rate)
        preferred_directions.append(preferred_direction[0])

    spatial_firing['max_firing_rate_hd'] = np.array(max_firing_rates) / 1000  # Hz
    spatial_firing['preferred_HD'] = preferred_directions
    return spatial_firing


def calculate_hd_score(spatial_firing):
    hd_scores = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        hd_hist = spatial_firing.hd_spike_histogram[cluster].copy()
        angles = np.linspace(-179, 180, 360)
        angles_rad = angles*np.pi/180
        dy = np.sin(angles_rad)
        dx = np.cos(angles_rad)

        totx = sum(dx * hd_hist)/sum(hd_hist)
        toty = sum(dy * hd_hist)/sum(hd_hist)
        r = np.sqrt(totx*totx + toty*toty)
        hd_scores.append(r)
    spatial_firing['hd_score'] = np.array(hd_scores)
    return spatial_firing


'''
p is the probability of obtaining two samples this different from the same distribution
stats is the raw test statistic
'''


def compare_hd_distributions_in_cluster_to_session(session_angles, cluster_angles):
    stats, p = kuiper.kuiper_two(session_angles, cluster_angles)
    return stats, p


def process_hd_data(spatial_firing, spatial_data, prm):
    print('I will process head-direction data now.')
    angles_whole_session = (np.array(spatial_data.hd) + 180) * np.pi / 180
    hd_histogram = get_hd_histogram(angles_whole_session)
    hd_histogram /= prm.get_sampling_rate()

    hd_spike_histograms = []
    hd_p_values = []
    hd_stat = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        angles_spike = (np.array(spatial_firing.hd[cluster]) + 180) * np.pi / 180
        stats, p = compare_hd_distributions_in_cluster_to_session(angles_whole_session, angles_spike)
        hd_p_values.append(p)
        hd_stat.append(stats)

        hd_spike_histogram = get_hd_histogram(angles_spike)
        hd_spike_histogram = hd_spike_histogram / hd_histogram
        hd_spike_histograms.append(hd_spike_histogram)

    spatial_firing['hd_spike_histogram'] = hd_spike_histograms
    spatial_firing['hd_p'] = hd_p_values
    spatial_firing['hd_stat'] = hd_stat
    spatial_firing = get_max_firing_rate(spatial_firing)
    spatial_firing = calculate_hd_score(spatial_firing)
    return hd_histogram, spatial_firing


# get HD data for a specific bin of the rate map
def get_indices_for_bin(bin_in_field, spatial_data, prm):
    bin_size_pixels = PostSorting.open_field_firing_maps.get_bin_size(prm)
    bin_x = bin_in_field[0]
    bin_x_left_pixels = bin_x * bin_size_pixels
    bin_x_right_pixels = bin_x * (bin_size_pixels + 1)
    bin_y = bin_in_field[1]
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
        inside_bin = get_indices_for_bin(bin_in_field, spatial_data, prm)
        hd = inside_bin.hd.values
        hd_in_field.extend(hd)

    return hd_in_field


# return array of HD in subfield when cell fired for cluster
def get_hd_in_firing_rate_bins_for_cluster(spatial_firing, rate_map_indices, cluster, prm):
    cluster_id = np.arange(len(spatial_firing.firing_times[cluster]))
    spatial_firing_cluster = pd.DataFrame(cluster_id)
    spatial_firing_cluster['x'] = spatial_firing.position_x_pixels[cluster]
    spatial_firing_cluster['y'] = spatial_firing.position_y_pixels[cluster]
    spatial_firing_cluster['hd'] = spatial_firing.hd[cluster]
    hd_in_field = get_hd_in_field(rate_map_indices, spatial_firing_cluster, prm)
    hd_in_field = (np.array(hd_in_field) + 180) * np.pi / 180
    return hd_in_field


# return array of HD angles in subfield when from the whole session
def get_hd_in_firing_rate_bins_for_session(spatial_data, rate_map_indices, prm):
    spatial_data_field = pd.DataFrame()
    spatial_data_field['x'] = spatial_data.position_x_pixels
    spatial_data_field['y'] = spatial_data.position_y_pixels
    spatial_data_field['hd'] = spatial_data.hd
    hd_in_field = get_hd_in_field(rate_map_indices, spatial_data_field, prm)
    hd_in_field = (np.array(hd_in_field) + 180) * np.pi / 180
    return hd_in_field


def main():
    pass

if __name__ == '__main__':
    main()