import array_utility
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import plot_utility
import scipy.ndimage
import scipy.stats


def calculate_speed_score(position, spatial_firing, sigma=8, sampling_rate_conversion=30000):
    speed = scipy.ndimage.filters.gaussian_filter(position.speed, sigma)
    speed_scores = []
    speed_score_ps = []
    for index, cell in spatial_firing.iterrows():
        firing_times = cell.firing_times
        firing_hist, edges = np.histogram(firing_times, bins=len(speed), range=(0, max(position.synced_time) * sampling_rate_conversion))
        smooth_hist = scipy.ndimage.filters.gaussian_filter(firing_hist.astype(float), sigma)
        speed, smooth_hist = array_utility.remove_nans_from_both_arrays(speed, smooth_hist)
        speed_score, p = scipy.stats.pearsonr(speed, smooth_hist)
        speed_scores.append(speed_score)
        speed_score_ps.append(p)
    spatial_firing['speed_score'] = speed_scores
    spatial_firing['speed_score_p_values'] = speed_score_ps

    return spatial_firing


def calculate_median_for_scatter(x, y):
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df_median = df.groupby('x')['y'].median()
    smooth_median = scipy.ndimage.gaussian_filter1d(df_median, sigma=1000)
    return smooth_median


def calculate_median_for_scatter_binned(x, y):
    bin_size = 6
    step_size = 2
    number_of_bins = int((max(x) - min(x)) / 2)

    median_x = []
    median_y = []
    percentile_25 = []
    percentile_75 = []
    for bin in range(number_of_bins):
        median_x.append(bin * step_size + bin_size/2)
        data_in_bin = np.take(y, np.where((bin * step_size < x) & (x < bin * step_size + bin_size)))
        med_y = np.median(data_in_bin)
        median_y.append(med_y)
        percentile_25.append(np.percentile(data_in_bin, 25))
        percentile_75.append(np.percentile(data_in_bin, 75))

    return median_x, median_y, percentile_25, percentile_75


def plot_speed_scores(position, spatial_firing, sigma, sampling_rate_conversion, save_path):
    speed = scipy.ndimage.filters.gaussian_filter(position.speed, sigma)
    for index, cell in spatial_firing.iterrows():
        firing_times = cell.firing_times
        firing_hist, edges = np.histogram(firing_times, bins=len(speed), range=(0, max(position.synced_time) * sampling_rate_conversion))
        smooth_hist = scipy.ndimage.filters.gaussian_filter(firing_hist.astype(float), sigma)
        speed, smooth_hist = array_utility.remove_nans_from_both_arrays(speed, smooth_hist)
        # speed = speed[::10]
        # smooth_hist = smooth_hist[::10]
        median_x, median_y, percentile_25, percentile_75 = calculate_median_for_scatter_binned(speed, smooth_hist)
        plt.cla()
        fig, ax = plt.subplots()
        ax = plot_utility.format_bar_chart(ax, 'Speed (cm/s)', 'Firing rate (Hz)')  # todo check if it's hz
        plt.scatter(speed[::10], smooth_hist[::10], color='gray', alpha=0.7)
        plt.plot(median_x, percentile_25, color='black', linewidth=5)
        plt.plot(median_x, percentile_75, color='black', linewidth=5)
        plt.scatter(median_x, median_y, color='black', s=100)
        plt.title('speed score: ' + str(np.round(cell.speed_score, 4)))
        plt.xlim(0, 50)
        plt.ylim(0, None)
        plt.savefig(save_path + cell.session_id + str(cell.cluster_id) + '_speed.png')
        plt.close()





