import numpy as np
import matplotlib.pylab as plt


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
    end = edges_result[window:len(edges_result)/2]
    beginning = edges_result[len(edges_result)/2:-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out


def get_hd_histogram(angles):
    theta = np.linspace(0, 2*np.pi, 360)  # x axis
    binned_hd, _, _ = plt.hist(angles, theta)


def process_hd_data(spatial_firing, spatial_data):
    cluster = 5
    angles = (np.array(spatial_firing.hd[cluster]) + 180) * np.pi / 180
    get_hd_histogram(angles)


def main():
    array_in = [1, 2, 3, 4, 5, 6]
    window = 3
    get_rolling_sum(array_in, window)

    array_in = [3, 4, 5, 8, 11, 1, 3, 5]
    window = 5
    get_rolling_sum(array_in, window)


if __name__ == '__main__':
    main()