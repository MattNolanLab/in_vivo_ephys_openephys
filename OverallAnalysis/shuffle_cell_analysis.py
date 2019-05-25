import glob
import matplotlib.pylab as plt
import numpy as np
import os
import OverallAnalysis.false_positives
import OverallAnalysis.folder_path_settings
import pandas as pd
import plot_utility
import PostSorting.open_field_grid_cells
import scipy
from scipy import stats
import shutil
from statsmodels.sandbox.stats.multicomp import multipletests

local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/shuffled_analysis_cell/'
local_path_mouse = local_path + 'all_mice_df.pkl'
local_path_rat = local_path + 'all_rats_df.pkl'

server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()


def format_bar_chart(ax):
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Head direction (deg)', fontsize=30)
    ax.set_ylabel('Frequency (Hz)', fontsize=30)
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
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'number_of_spikes', 'mean_firing_rate', 'hd_score', 'position_x', 'position_y', 'hd', 'firing_maps']].copy()
                spatial_firing['trajectory_hd'] = [position.hd] * len(spatial_firing)
                spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)
                spatial_firing_data = spatial_firing_data.append(spatial_firing)
                print(spatial_firing_data.head())

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def add_combined_id_to_df(spatial_firing):
    animal_ids = [session_id.split('_')[0] for session_id in spatial_firing.session_id.values]
    dates = [session_id.split('_')[1] for session_id in spatial_firing.session_id.values]
    if 'tetrode' in spatial_firing:
        tetrode = spatial_firing.tetrode.values
        cluster = spatial_firing.cluster_id.values

        combined_ids = []
        for cell in range(len(spatial_firing)):
            id = animal_ids[cell] + '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
            combined_ids.append(id)
        spatial_firing['false_positive_id'] = combined_ids
    else:
        cluster = spatial_firing.cluster_id.values
        combined_ids = []
        for cell in range(len(spatial_firing)):
            id = animal_ids[cell] + '-' + dates[cell] + '-Cluster-' + str(cluster[cell])
            combined_ids.append(id)
        spatial_firing['false_positive_id'] = combined_ids

    return spatial_firing


def tag_false_positives(spatial_firing):
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(local_path + 'false_positives_all.txt')
    spatial_firing = add_combined_id_to_df(spatial_firing)
    spatial_firing['false_positive'] = spatial_firing['false_positive_id'].isin(list_of_false_positives)
    return spatial_firing


def add_mean_and_std_to_df(spatial_firing, sampling_rate_video, number_of_bins=20):
    shuffled_means = []
    shuffled_stdevs = []
    real_data_hz_all = []
    time_spent_in_bins_all = []
    histograms_hz_all_shuffled = []
    for index, cell in spatial_firing.iterrows():
        shuffled_histograms = cell['shuffled_data']
        cell_spikes_hd = np.asanyarray(cell['hd'])
        cell_spikes_hd = cell_spikes_hd[~np.isnan(cell_spikes_hd)]  # real hd when the cell fired
        cell_session_hd = np.asanyarray(cell['trajectory_hd'])  # hd from the whole session in field
        cell_session_hd = cell_session_hd[~np.isnan(cell_session_hd)]
        time_spent_in_bins = np.histogram(cell_session_hd, bins=number_of_bins)[0]
        time_spent_in_bins_all.append(time_spent_in_bins)
        shuffled_histograms_hz = shuffled_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        histograms_hz_all_shuffled.append(shuffled_histograms_hz)
        mean_shuffled = np.mean(shuffled_histograms_hz, axis=0)
        shuffled_means.append(mean_shuffled)
        std_shuffled = np.std(shuffled_histograms_hz, axis=0)
        shuffled_stdevs.append(std_shuffled)
        real_data_hz = np.histogram(cell_spikes_hd, bins=number_of_bins)[0] * sampling_rate_video / time_spent_in_bins
        real_data_hz_all.append(real_data_hz)
    spatial_firing['shuffled_means'] = shuffled_means
    spatial_firing['shuffled_std'] = shuffled_stdevs
    spatial_firing['hd_histogram_real_data_hz'] = real_data_hz_all
    spatial_firing['time_spent_in_bins'] = time_spent_in_bins_all
    spatial_firing['shuffled_histograms_hz'] = histograms_hz_all_shuffled
    return spatial_firing


def add_percentile_values_to_df(spatial_firing, sampling_rate_video, number_of_bins=20):
    percentile_values_95_all = []
    percentile_values_5_all = []
    error_bar_up_all = []
    error_bar_down_all = []
    for index, cell in spatial_firing.iterrows():
        shuffled_cell_histograms = cell['shuffled_data']
        time_spent_in_bins = cell.time_spent_in_bins  # based on trajectory
        shuffled_histograms_hz = shuffled_cell_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        percentile_value_shuffled_95 = np.percentile(shuffled_histograms_hz, 95, axis=0)
        percentile_values_95_all.append(percentile_value_shuffled_95)
        percentile_value_shuffled_5 = np.percentile(shuffled_histograms_hz, 5, axis=0)
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
        diff_cell = (cell.shuffled_percentile_threshold_95 < cell.hd_histogram_real_data_hz) + (cell.shuffled_percentile_threshold_5 > cell.hd_histogram_real_data_hz)  # this is a pairwise OR on the binary arrays
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
            diff_cell = (cell.shuffled_percentile_threshold_95 < cell.shuffled_histograms_hz[shuffle]) + (cell.shuffled_percentile_threshold_5 > cell.shuffled_histograms_hz[shuffle])  # this is a pairwise OR on the binary arrays
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
        shuffled_histograms = cell['shuffled_data']
        time_spent_in_bins = cell.time_spent_in_bins
        shuffled_data_normalized = shuffled_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
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
        shuffled_histograms = cell['shuffled_data']
        time_spent_in_bins = cell.time_spent_in_bins
        shuffled_data_normalized = shuffled_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        percentiles_of_observed_bars = np.empty(number_of_bars)
        percentiles_of_observed_bars[:] = np.nan
        for bar in range(number_of_bars):
            observed_data = cell.hd_histogram_real_data_hz[bar]
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


def plot_bar_chart_for_cells(spatial_firing, path, animal):
    for index, cell in spatial_firing.iterrows():
        mean = cell['shuffled_means']
        std = cell['shuffled_std']
        cell_spikes_hd = np.array(cell['hd'])
        shuffled_histograms_hz = cell['shuffled_histograms_hz']
        x_pos = np.arange(shuffled_histograms_hz.shape[1])
        fig, ax = plt.subplots()
        ax = format_bar_chart(ax)
        ax.bar(x_pos, mean, yerr=std*2, align='center', alpha=0.7, color='black', ecolor='grey', capsize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        plt.scatter(x_pos, cell.hd_histogram_real_data_hz, marker='o', color='red', s=40)
        plt.savefig(path + 'shuffle_analysis_' + animal + '/' + animal + str(cell['session_id']) + str(cell['cluster_id']) + str(index) + '_SD')
        plt.close()


def plot_bar_chart_for_cells_percentile_error_bar(spatial_firing, path, animal):
    for index, cell in spatial_firing.iterrows():
        mean = cell['shuffled_means']
        percentile_95 = cell['error_bar_95']
        percentile_5 = cell['error_bar_5']
        shuffled_histograms_hz = cell['shuffled_histograms_hz']
        x_pos = np.arange(shuffled_histograms_hz.shape[1])
        fig, ax = plt.subplots()
        ax = format_bar_chart(ax)
        ax.errorbar(x_pos, mean, yerr=[percentile_5, percentile_95], alpha=0.7, color='black', ecolor='grey', capsize=10, fmt='o', markersize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        plt.scatter(x_pos, cell.hd_histogram_real_data_hz, marker='o', color='navy', s=40)
        plt.savefig(path + 'shuffle_analysis_' + animal + '/' + animal + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile')
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

    if os.path.exists(local_path + 'shuffle_analysis_' + animal) is True:
        shutil.rmtree(local_path + 'shuffle_analysis_' + animal)
    os.makedirs(local_path + 'shuffle_analysis_' + animal)

    shuffled_histograms_all = []
    for index, cell in spatial_firing.iterrows():
        print('I will shuffle data.')
        shuffled_histograms = np.zeros((number_of_times_to_shuffle, number_of_bins))
        shuffle_indices = get_random_indices_for_shuffle(cell, number_of_times_to_shuffle)
        for shuffle in range(number_of_times_to_shuffle):
            shuffled_hd = cell['trajectory_hd'][shuffle_indices[shuffle]]
            shuffled_hd = (shuffled_hd + 180) * np.pi / 180
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
    if 'number_of_different_bins_shuffled_corrected_p' in spatial_firing:
        return spatial_firing
    print('Analyze shuffled data.')
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
    plot_bar_chart_for_cells(spatial_firing, save_path, animal)
    plot_bar_chart_for_cells_percentile_error_bar(spatial_firing, save_path, animal)
    if animal == 'mouse':
        spatial_firing.to_pickle(local_path_mouse)
    if animal == 'rat':
        spatial_firing.to_pickle(local_path_rat)
    return spatial_firing


def find_tail_of_shuffled_distribution_of_rejects(shuffled_field_data):
    number_of_rejects = shuffled_field_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for field in number_of_rejects:
        flat_shuffled.extend(field)
    tail = max(flat_shuffled)
    percentile_95 = np.percentile(flat_shuffled, 95)
    percentile_99 = np.percentile(flat_shuffled, 99)
    return tail, percentile_95, percentile_99


def plot_histogram_of_number_of_rejected_bars(shuffled_field_data, animal='mouse'):
    number_of_rejects = shuffled_field_data.number_of_different_bins
    fig, ax = plt.subplots()
    plt.hist(number_of_rejects)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim(0, 20.5)
    ax.set_xlabel('Rejected bars / cell', size=30)
    ax.set_ylabel('Proportion', size=30)
    plt.savefig(local_path + 'distribution_of_rejects_' + animal + '.png', bbox_inches="tight")
    plt.close()


def plot_histogram_of_number_of_rejected_bars_shuffled(shuffled_data, animal='mouse'):
    number_of_rejects = shuffled_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for cell in number_of_rejects:
        flat_shuffled.extend(cell)
    fig, ax = plt.subplots()
    plt.hist(flat_shuffled, color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Rejected bars / cell', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_xlim(0, 20.5)
    plt.savefig(local_path + '/distribution_of_rejects_shuffled' + animal + '.png', bbox_inches="tight")
    plt.close()


def make_combined_plot_of_distributions(shuffled_data, tag='grid'):
    tail, percentile_95, percentile_99 = find_tail_of_shuffled_distribution_of_rejects(shuffled_data)

    number_of_rejects_shuffled = shuffled_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for cell in number_of_rejects_shuffled:
        flat_shuffled.extend(cell)
    fig, ax = plt.subplots()
    plt.hist(flat_shuffled, normed=True, color='black', alpha=0.5)

    number_of_rejects_real = shuffled_data.number_of_different_bins
    plt.hist(number_of_rejects_real, normed=True, color='navy', alpha=0.5)

    # plt.axvline(x=tail, color='red', alpha=0.5, linestyle='dashed')
    # plt.axvline(x=percentile_95, color='red', alpha=0.5, linestyle='dashed')
    # plt.axvline(x=percentile_99, color='red', alpha=0.5, linestyle='dashed')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Rejected bars / cell', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_xlim(0, 20.5)
    plt.savefig(local_path + 'distribution_of_rejects_combined_all_' + tag + '.png', bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    plt.yticks([0, 1])
    ax = plot_utility.format_bar_chart(ax, 'Rejected bars / cell', 'Cumulative probability')
    values, base = np.histogram(flat_shuffled, bins=40)
    cumulative = np.cumsum(values / len(flat_shuffled))
    plt.plot(base[:-1], cumulative, c='gray', linewidth=5)

    values, base = np.histogram(number_of_rejects_real, bins=40)
    cumulative = np.cumsum(values / len(number_of_rejects_real))
    plt.plot(base[:-1], cumulative, c='navy', linewidth=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim(0, 20.5)
    ax.set_xlabel('Rejected bars / cell', size=30)
    ax.set_ylabel('Cumulative probability', size=30)
    plt.savefig(local_path + 'distribution_of_rejects_' + tag + '_cumulative.png', bbox_inches="tight")
    plt.close()


def plot_number_of_significant_p_values(spatial_firing, type='bh'):
    if type == 'bh':
        number_of_significant_p_values = spatial_firing.number_of_different_bins_bh
    else:
        number_of_significant_p_values = spatial_firing.number_of_different_bins_holm

    fig, ax = plt.subplots()
    plt.hist(number_of_significant_p_values, normed='True', color='navy', alpha=0.5)
    flat_shuffled = []
    for cell in spatial_firing.number_of_different_bins_shuffled_corrected_p:
        flat_shuffled.extend(cell)
    plt.hist(flat_shuffled, normed='True', color='gray', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Significant bars / cell', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_ylim(0, 0.2)
    ax.set_xlim(0, 20.5)
    plt.savefig(local_path + 'distribution_of_rejects_significant_p_ ' + type + '.png', bbox_inches = "tight")
    plt.close()

    fig, ax = plt.subplots()
    plt.xscale('log')
    plt.yticks([0, 1])
    ax = plot_utility.format_bar_chart(ax, 'Rejected bars / cell', 'Cumulative probability')
    values, base = np.histogram(flat_shuffled, bins=40)
    cumulative = np.cumsum(values / len(flat_shuffled))
    plt.plot(base[:-1], cumulative, c='gray', linewidth=5)

    values, base = np.histogram(number_of_significant_p_values, bins=40)
    cumulative = np.cumsum(values / len(number_of_significant_p_values))
    plt.plot(base[:-1], cumulative, c='navy', linewidth=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim(0, 20.5)
    ax.set_xlabel('Rejected bars / cell', size=30)
    ax.set_ylabel('Cumulative probability', size=30)
    plt.savefig(local_path + 'distribution_of_rejects_signigicant_p' + type + '_cumulative.png', bbox_inches="tight")
    plt.close()


def compare_distributions(x, y):
    stat, p = scipy.stats.mannwhitneyu(x, y)
    return p


def compare_shuffled_to_real_data_mw_test(spatial_firing, analysis_type='bh'):
    if analysis_type == 'bh':
        flat_shuffled = []
        for cell in spatial_firing.number_of_different_bins_shuffled_corrected_p:
            flat_shuffled.extend(cell)
            p_bh = compare_distributions(spatial_firing.number_of_different_bins_bh, flat_shuffled)
            print('Number of cells: ' + str(len(spatial_firing)))
            print('p value for comparing shuffled distribution to B-H corrected p values: ' + str(p_bh))
            return p_bh

    if analysis_type == 'percentile':
        flat_shuffled = []
        for cell in spatial_firing.number_of_different_bins_shuffled:
            flat_shuffled.extend(cell)
            p_percentile = compare_distributions(spatial_firing.number_of_different_bins, flat_shuffled)
            print('p value for comparing shuffled distribution to percentile thresholded p values: ' + str(p_percentile))
            return p_percentile


def plot_distributions_for_shuffled_vs_real_cells(shuffled_spatial_firing_data, tag='grid', animal='mouse'):
    plot_histogram_of_number_of_rejected_bars(shuffled_spatial_firing_data, animal)
    plot_histogram_of_number_of_rejected_bars_shuffled(shuffled_spatial_firing_data, animal)
    plot_number_of_significant_p_values(shuffled_spatial_firing_data, type='bh_' + tag + '_' + animal)
    plot_number_of_significant_p_values(shuffled_spatial_firing_data, type='holm_' + tag + '_' + animal)
    make_combined_plot_of_distributions(shuffled_spatial_firing_data, tag=tag + '_' + animal)


def process_data(spatial_firing, sampling_rate_video, animal='mouse'):
    spatial_firing = shuffle_data(spatial_firing, 20, number_of_times_to_shuffle=1000, animal=animal)
    spatial_firing = analyze_shuffled_data(spatial_firing, local_path, sampling_rate_video, animal, number_of_bins=20)
    print('I finished the shuffled analysis on ' + animal + ' data.')
    if animal == 'mouse':
        spatial_firing = tag_false_positives(spatial_firing)
    else:
        spatial_firing['false_positive'] = False

    good_cell = spatial_firing.false_positive == False
    grid = spatial_firing.grid_score >= 0.4
    hd = spatial_firing.hd_score >= 0.5
    not_classified = np.logical_and(np.logical_not(grid), np.logical_not(hd))
    hd_cells = np.logical_and(np.logical_not(grid), hd)
    grid_cells = np.logical_and(grid, np.logical_not(hd))

    shuffled_spatial_firing_grid = spatial_firing[grid_cells & good_cell]
    shuffled_spatial_firing_not_classified = spatial_firing[not_classified & good_cell]

    plot_distributions_for_shuffled_vs_real_cells(shuffled_spatial_firing_grid, 'grid', animal=animal)
    plot_distributions_for_shuffled_vs_real_cells(shuffled_spatial_firing_not_classified, 'not_classified', animal=animal)

    print(animal + ' data:')
    print('Grid cells:')
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_grid, analysis_type='bh')
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_grid, analysis_type='percentile')
    print('Not classified cells:')
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_not_classified, analysis_type='bh')
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_not_classified, analysis_type='percentile')


def main():
    spatial_firing_all_mice = load_data_frame_spatial_firing(local_path_mouse, server_path_mouse, spike_sorter='/MountainSort')
    spatial_firing_all_rats = load_data_frame_spatial_firing(local_path_rat, server_path_rat, spike_sorter='')
    process_data(spatial_firing_all_mice, 30, animal='mouse')
    process_data(spatial_firing_all_rats, 50, animal='rat')


if __name__ == '__main__':
    main()
