import glob
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
import shutil
from statsmodels.sandbox.stats.multicomp import multipletests


local_path_mouse = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/shuffled_analysis_cell/all_mice_df.pkl'
local_path_rat = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/shuffled_analysis_cell/all_rats_df.pkl'
analysis_path = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/shuffled_analysis_cell/'

server_path_mouse = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/Open_field_opto_tagging_p038/'
server_path_rat = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/grid_field_analysis/moser_data/Sargolini/all_data/'


def format_bar_chart(ax):
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Head direction [deg]', fontsize=30)
    ax.set_ylabel('Frequency [Hz]', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    return ax


def load_data_frame_spatial_firing(output_path, server_path, spike_sorter='/MountainSort'):
    if os.path.exists(output_path):
        spatial_firing = pd.read_pickle(output_path)
        return spatial_firing
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        firing_data_frame_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        if os.path.exists(firing_data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(firing_data_frame_path)
            position = pd.read_pickle(position_path)

            if 'position_x' in spatial_firing:
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'number_of_spikes', 'mean_firing_rate', 'hd_score', 'position_x', 'position_y', 'hd']].copy()
                spatial_firing['trajectory_hd'] = [position.hd] * len(spatial_firing)
                spatial_firing_data = spatial_firing_data.append(spatial_firing)
                print(spatial_firing_data.head())

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def add_mean_and_std_to_df(spatial_firing, sampling_rate_video, number_of_bins=20):
    shuffled_means = []
    shuffled_stdevs = []
    real_data_hz_all = []
    time_spent_in_bins_all = []
    histograms_hz_all = []
    for index, cell in spatial_firing.iterrows():
        cell_histograms = cell['shuffled_data']
        cell_spikes_hd = np.asanyarray(cell['hd'])
        cell_spikes_hd = cell_spikes_hd[~np.isnan(cell_spikes_hd)]  # real hd when the cell fired
        cell_session_hd = np.asanyarray(cell['trajectory_hd'])  # hd from the whole session in field
        cell_session_hd = cell_session_hd[~np.isnan(cell_session_hd)]
        time_spent_in_bins = np.histogram(cell_session_hd, bins=number_of_bins)[0]
        time_spent_in_bins_all.append(time_spent_in_bins)
        cell_histograms_hz = cell_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        histograms_hz_all.append(cell_histograms_hz)
        mean_shuffled = np.mean(cell_histograms_hz, axis=0)
        shuffled_means.append(mean_shuffled)
        std_shuffled = np.std(cell_histograms_hz, axis=0)
        shuffled_stdevs.append(std_shuffled)

        real_data_hz = np.histogram(cell_spikes_hd, bins=number_of_bins)[0] * sampling_rate_video / time_spent_in_bins
        real_data_hz_all.append(real_data_hz)
    spatial_firing['shuffled_means'] = shuffled_means
    spatial_firing['shuffled_std'] = shuffled_stdevs
    spatial_firing['hd_histogram_real_data'] = real_data_hz_all
    spatial_firing['time_spent_in_bins'] = time_spent_in_bins_all
    spatial_firing['cell_histograms_hz'] = histograms_hz_all
    return spatial_firing


def add_percentile_values_to_df(spatial_firing, sampling_rate_video, number_of_bins=20):
    percentile_values_95_all = []
    percentile_values_5_all = []
    error_bar_up_all = []
    error_bar_down_all = []
    for index, cell in spatial_firing.iterrows():
        shuffled_cell_histograms = cell['shuffled_data']
        session_hd = np.asanyarray(cell['trajectory_hd'])  # hd from the whole session
        session_hd = session_hd[~np.isnan(session_hd)]
        time_spent_in_bins = np.histogram(session_hd, bins=number_of_bins)[0]
        cell_histograms_hz = shuffled_cell_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        percentile_value_shuffled_95 = np.percentile(cell_histograms_hz, 95, axis=0)
        percentile_values_95_all.append(percentile_value_shuffled_95)
        percentile_value_shuffled_5 = np.percentile(cell_histograms_hz, 5, axis=0)
        percentile_values_5_all.append(percentile_value_shuffled_5)
        error_bar_up = percentile_value_shuffled_95 - cell.shuffled_means
        error_bar_down = cell.shuffled_means - percentile_value_shuffled_5
        error_bar_up_all.append(error_bar_up)
        error_bar_down_all.append(error_bar_down)
    spatial_firing['shuffled_percentile_threshold_95'] = percentile_values_95_all
    spatial_firing['shuffled_percentile_threshold_5'] = percentile_values_5_all
    spatial_firing['error_bar_95'] = error_bar_up_all
    spatial_firing['error_bar_5'] = error_bar_down_all
    return spatial_firing


# test whether real and shuffled data differ and add results (true/false for each bin) and number of diffs to data frame
def test_if_real_hd_differs_from_shuffled(spatial_firing):
    real_and_shuffled_data_differ_bin = []
    number_of_diff_bins = []
    for index, cell in spatial_firing.iterrows():
        # diff_field = np.abs(field.shuffled_means - field.hd_histogram_real_data) > field.shuffled_std * 2
        diff_cell = (cell.shuffled_percentile_threshold_95 < cell.hd_histogram_real_data) + (cell.shuffled_percentile_threshold_5 > cell.hd_histogram_real_data)  # this is a pairwise OR on the binary arrays
        number_of_diffs = diff_cell.sum()
        real_and_shuffled_data_differ_bin.append(diff_cell)
        number_of_diff_bins.append(number_of_diffs)
    spatial_firing['real_and_shuffled_data_differ_bin'] = real_and_shuffled_data_differ_bin
    spatial_firing['number_of_different_bins'] = number_of_diff_bins
    return spatial_firing


# this uses the p values that are based on the position of the real data relative to shuffled (corrected_
def count_number_of_significantly_different_bars_per_field(spatial_firing, significance_level=95, type='bh'):
    number_of_significant_p_values = []
    false_positive_ratio = (100 - significance_level) / 100
    for index, cell in spatial_firing.iterrows():
        # count significant p values
        if type == 'bh':
            number_of_significant_p_values_cell = (cell.p_values_corrected_bars_bh < false_positive_ratio).sum()
            number_of_significant_p_values.append(number_of_significant_p_values_cell)
        if type == 'holm':
            number_of_significant_p_values_cell = (cell.p_values_corrected_bars_holm < false_positive_ratio).sum()
            number_of_significant_p_values.append(number_of_significant_p_values_cell)
    field_name = 'number_of_different_bins_' + type
    spatial_firing[field_name] = number_of_significant_p_values
    return spatial_firing


# this is to find the null distribution of number of rejected null hypothesis based on the shuffled data
def test_if_shuffle_differs_from_other_shuffles(spatial_firing):
    number_of_shuffles = len(spatial_firing.shuffled_data.iloc[0])
    rejected_bins_all_shuffles = []
    for index, cell in spatial_firing.iterrows():
        rejects_cell = np.empty(number_of_shuffles)
        rejects_cell[:] = np.nan
        for shuffle in range(number_of_shuffles):
            diff_cell = (cell.shuffled_percentile_threshold_95 < cell.cell_histograms_hz[shuffle]) + (cell.shuffled_percentile_threshold_5 > cell.cell_histograms_hz[shuffle])  # this is a pairwise OR on the binary arrays
            number_of_diffs = diff_cell.sum()
            rejects_cell[shuffle] = number_of_diffs
        rejected_bins_all_shuffles.append(rejects_cell)
    spatial_firing['number_of_different_bins_shuffled'] = rejected_bins_all_shuffles
    return spatial_firing


# this is to find the null distribution of number of rejected null hypothesis based on the shuffled data
# perform B/H analysis on each shuffle and count rejects
def test_if_shuffle_differs_from_other_shuffles_corrected_p_values(spatial_firing, sampling_rate_video, number_of_bars=20):
    number_of_shuffles = len(spatial_firing.shuffled_data.iloc[0])
    rejected_bins_all_shuffles = []
    for index, cell in spatial_firing.iterrows():
        cell_histograms = cell['shuffled_data']
        cell_session_hd = np.asanyarray(cell['trajectory_hd'])  # hd from the whole session in field
        cell_session_hd = cell_session_hd[~np.isnan(cell_session_hd)]
        time_spent_in_bins = np.histogram(cell_session_hd, bins=number_of_bars)[0]
        shuffled_data_normalized = cell_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        rejects_cell = np.empty(number_of_shuffles)
        rejects_cell[:] = np.nan
        percentile_observed_data_bars = []
        for shuffle in range(number_of_shuffles):
            percentiles_of_observed_bars = np.empty(number_of_bars)
            percentiles_of_observed_bars[:] = np.nan
            for bar in range(number_of_bars):
                observed_data = shuffled_data_normalized[shuffle][bar]
                shuffled_data = shuffled_data_normalized[:, bar]
                percentile_of_observed_data = stats.percentileofscore(shuffled_data, observed_data)
                percentiles_of_observed_bars[bar] = percentile_of_observed_data
            percentile_observed_data_bars.append(percentiles_of_observed_bars)  # percentile of shuffle relative to all other shuffles
            # convert percentile to p value
            percentiles_of_observed_bars[percentiles_of_observed_bars > 50] = 100 - percentiles_of_observed_bars[percentiles_of_observed_bars > 50]
            # correct p values (B/H)
            reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(percentiles_of_observed_bars, alpha=0.05, method='fdr_bh')
            # count significant bars and put this number in df
            number_of_rejects = reject.sum()
            rejects_cell[shuffle] = number_of_rejects
        rejected_bins_all_shuffles.append(rejects_cell)
    spatial_firing['number_of_different_bins_shuffled_corrected_p'] = rejected_bins_all_shuffles
    return spatial_firing


# calculate percentile of real data relative to shuffled for each bar
def calculate_percentile_of_observed_data(spatial_firing, sampling_rate_video, number_of_bars=20):
    percentile_observed_data_bars = []
    for index, cell in spatial_firing.iterrows():
        cell_histograms = cell['shuffled_data']
        cell_session_hd = np.asanyarray(cell['trajectory_hd'])  # hd from the whole session in field
        cell_session_hd = cell_session_hd[~np.isnan(cell_session_hd)]
        time_spent_in_bins = np.histogram(cell_session_hd, bins=number_of_bars)[0]
        shuffled_data_normalized = cell_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        percentiles_of_observed_bars = np.empty(number_of_bars)
        percentiles_of_observed_bars[:] = np.nan
        for bar in range(number_of_bars):
            observed_data = cell.hd_histogram_real_data[bar]
            shuffled_data = shuffled_data_normalized[:, bar]
            percentile_of_observed_data = stats.percentileofscore(shuffled_data, observed_data)
            percentiles_of_observed_bars[bar] = percentile_of_observed_data
        percentile_observed_data_bars.append(percentiles_of_observed_bars)
    spatial_firing['percentile_of_observed_data'] = percentile_observed_data_bars
    return spatial_firing


#  convert percentile to p value by subtracting the percentile from 100 when it is > than 50
def convert_percentile_to_p_value(spatial_firing):
    p_values = []
    for index, cell in spatial_firing.iterrows():
        percentile_values = cell.percentile_of_observed_data
        percentile_values[percentile_values > 50] = 100 - percentile_values[percentile_values > 50]
        p_values.append(percentile_values)
    spatial_firing['shuffle_p_values'] = p_values
    return spatial_firing


# perform Benjamini/Hochberg correction on p values calculated from the percentile of observed data relative to shuffled
def calculate_corrected_p_values(spatial_firing, type='bh'):
    corrected_p_values = []
    for index, cell in spatial_firing.iterrows():
        p_values = cell.shuffle_p_values
        if type == 'bh':
            reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values, alpha=0.05, method='fdr_bh')
            corrected_p_values.append(pvals_corrected)
        if type == 'holm':
            reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values, alpha=0.05, method='holm')
            corrected_p_values.append(pvals_corrected)

    field_name = 'p_values_corrected_bars_' + type
    spatial_firing[field_name] = corrected_p_values
    return spatial_firing


def plot_bar_chart_for_cells(spatial_firing, path, sampling_rate_video, animal):
    for index, cell in spatial_firing.iterrows():
        mean = cell['shuffled_means']
        std = cell['shuffled_std']
        cell_spikes_hd = np.array(cell['hd'])
        cell_spikes_hd = cell_spikes_hd[~np.isnan(cell_spikes_hd)]
        time_spent_in_bins = cell['time_spent_in_bins']
        cell_histograms_hz = cell['cell_histograms_hz']
        x_pos = np.arange(cell_histograms_hz.shape[1])
        fig, ax = plt.subplots()
        ax = format_bar_chart(ax)
        ax.bar(x_pos, mean, yerr=std*2, align='center', alpha=0.7, color='black', ecolor='grey', capsize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        real_data_hz = np.histogram(cell_spikes_hd, bins=20)[0] * sampling_rate_video / time_spent_in_bins
        plt.scatter(x_pos, real_data_hz, marker='o', color='red', s=40)
        plt.savefig(path + 'shuffle_analysis/' + animal + str(cell['session_id']) + str(cell['cluster_id']) + str(index) + '_SD')
        plt.close()


def plot_bar_chart_for_cells_percentile_error_bar(spatial_firing, path, sampling_rate_video, animal):
    for index, cell in spatial_firing.iterrows():
        mean = cell['shuffled_means']
        percentile_95 = cell['error_bar_95']
        percentile_5 = cell['error_bar_5']
        cell_spikes_hd = np.array(cell['hd'])
        cell_spikes_hd = cell_spikes_hd[~np.isnan(cell_spikes_hd)]
        time_spent_in_bins = cell['time_spent_in_bins']
        cell_histograms_hz = cell['cell_histograms_hz']
        x_pos = np.arange(cell_histograms_hz.shape[1])
        fig, ax = plt.subplots()
        ax = format_bar_chart(ax)
        ax.errorbar(x_pos, mean, yerr=[percentile_5, percentile_95], alpha=0.7, color='black', ecolor='grey', capsize=10, fmt='o', markersize=10)
        # ax.bar(x_pos, mean, yerr=[percentile_5, percentile_95], align='center', alpha=0.7, color='black', ecolor='grey', capsize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        real_data_hz = np.histogram(cell_spikes_hd, bins=20)[0] * sampling_rate_video / time_spent_in_bins
        plt.scatter(x_pos, real_data_hz, marker='o', color='red', s=40)
        plt.savefig(path + 'shuffle_analysis/' + animal + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile')
        plt.close()


def get_random_indices_for_shuffle(cell, number_of_times_to_shuffle):
    number_of_spikes_in_field = cell['number_of_spikes']
    length_of_recording = len(cell.trajectory_hd)
    shuffle_indices = np.random.randint(0, length_of_recording, size=(number_of_times_to_shuffle, number_of_spikes_in_field))
    return shuffle_indices


# add shuffled data to data frame as a new column for each cell
def shuffle_data(spatial_firing, number_of_bins, number_of_times_to_shuffle=1000, animal='mouse'):
    if 'shuffled_data' in spatial_firing:
        return spatial_firing

    if os.path.exists(analysis_path + 'shuffle_analysis') is True:
        shutil.rmtree(analysis_path + 'shuffle_analysis')
    os.makedirs(analysis_path + 'shuffle_analysis')

    shuffled_histograms_all = []
    for index, cell in spatial_firing.iterrows():
        print('I will shuffle data.')
        shuffled_histograms = np.zeros((number_of_times_to_shuffle, number_of_bins))
        shuffle_indices = get_random_indices_for_shuffle(cell, number_of_times_to_shuffle)
        for shuffle in range(number_of_times_to_shuffle):
            shuffled_hd = cell['trajectory_hd'][shuffle_indices[shuffle]]
            hist, bin_edges = np.histogram(shuffled_hd, bins=number_of_bins, range=(0, 6.28))  # from 0 to 2pi
            shuffled_histograms[shuffle, :] = hist
        shuffled_histograms_all.append(shuffled_histograms)
    spatial_firing['shuffled_data'] = shuffled_histograms_all
    if animal == 'mouse':
        spatial_firing.to_pickle(local_path_mouse)

    if animal == 'rat':
        spatial_firing.to_pickle(local_path_rat)

    return spatial_firing


def analyze_shuffled_data(spatial_firing, save_path, sampling_rate_video, animal, number_of_bins=20):
    spatial_firing = add_mean_and_std_to_df(spatial_firing, sampling_rate_video, number_of_bins)
    spatial_firing = add_percentile_values_to_df(spatial_firing, sampling_rate_video, number_of_bins=20)
    spatial_firing = test_if_real_hd_differs_from_shuffled(spatial_firing)  # is the observed data within 95th percentile of the shuffled?
    spatial_firing = test_if_shuffle_differs_from_other_shuffles(spatial_firing)

    spatial_firing = calculate_percentile_of_observed_data(spatial_firing, sampling_rate_video, number_of_bins)  # this is relative to shuffled data
    # field_data = calculate_percentile_of_shuffled_data(field_data, number_of_bars=20)
    spatial_firing = convert_percentile_to_p_value(spatial_firing)  # this is needed to make it 2 tailed so diffs are picked up both ways
    spatial_firing = calculate_corrected_p_values(spatial_firing, type='bh')  # BH correction on p values from previous function
    spatial_firing = calculate_corrected_p_values(spatial_firing, type='holm')  # Holm correction on p values from previous function
    spatial_firing = count_number_of_significantly_different_bars_per_field(spatial_firing, significance_level=95, type='bh')
    spatial_firing = count_number_of_significantly_different_bars_per_field(spatial_firing, significance_level=95, type='holm')
    spatial_firing = test_if_shuffle_differs_from_other_shuffles_corrected_p_values(spatial_firing, sampling_rate_video, number_of_bars=20)
    plot_bar_chart_for_cells(spatial_firing, save_path, sampling_rate_video, animal)
    plot_bar_chart_for_cells_percentile_error_bar(spatial_firing, save_path, sampling_rate_video, animal)
    return spatial_firing


def process_data(spatial_firing, sampling_rate_video, animal='mouse'):
    spatial_firing = shuffle_data(spatial_firing, 20, number_of_times_to_shuffle=1000, animal=animal)
    spatial_firing = analyze_shuffled_data(spatial_firing, analysis_path, sampling_rate_video, animal, number_of_bins=20)


def main():
    spatial_firing_all_mice = load_data_frame_spatial_firing(local_path_mouse, server_path_mouse, spike_sorter='/MountainSort')
    spatial_firing_all_rats = load_data_frame_spatial_firing(local_path_rat, server_path_rat, spike_sorter='')
    process_data(spatial_firing_all_mice, 30, animal='mouse')
    process_data(spatial_firing_all_rats, 50, animal='rat')


if __name__ == '__main__':
    main()
