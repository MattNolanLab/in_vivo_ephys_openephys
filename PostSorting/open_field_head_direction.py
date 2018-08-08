import math
import matplotlib.pylab as plt
import numpy as np


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


def get_hd_in_firing_rate_bins_for_cluster(spatial_firing, rate_map_indices):
    hd_array = []



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