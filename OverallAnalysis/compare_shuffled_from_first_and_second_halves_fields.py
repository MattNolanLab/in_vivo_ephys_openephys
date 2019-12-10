import array_utility
import glob
import matplotlib.pylab as plt
import numpy as np
import os
import OverallAnalysis.compare_shuffled_from_first_and_second_halves
import OverallAnalysis.folder_path_settings
import OverallAnalysis.false_positives
import OverallAnalysis.open_field_firing_maps_processed_data
import OverallAnalysis.shuffle_field_analysis
import OverallAnalysis.shuffle_field_analysis_all_animals
import pandas as pd
import OverallAnalysis.shuffle_cell_analysis
import OverallAnalysis.shuffle_field_analysis
import PostSorting.compare_first_and_second_half
import PostSorting.open_field_firing_maps
import PostSorting.open_field_head_direction
import PostSorting.parameters
import scipy.stats
from scipy import signal

prm = PostSorting.parameters.Parameters()

local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/compare_first_and_second_shuffled_fields/'
local_path_mouse = local_path + 'all_mice_df.pkl'
local_path_mouse_down_sampled = local_path + 'all_mice_df_down_sampled.pkl'
local_path_rat = local_path + 'all_rats_df.pkl'

server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()


def add_cell_types_to_data_frame(spatial_firing):
    cell_type = []
    for index, cell in spatial_firing.iterrows():
        if cell.hd_score >= 0.5 and cell.grid_score >= 0.4:
            cell_type.append('conjunctive')
        elif cell.hd_score >= 0.5:
            cell_type.append('hd')
        elif cell.grid_score >= 0.4:
            cell_type.append('grid')
        else:
            cell_type.append('na')

    spatial_firing['cell type'] = cell_type

    return spatial_firing


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


def load_data(path):
    first_half_spatial_firing = None
    second_half_spatial_firing = None
    first_position = None
    second_position = None
    if os.path.exists(path + '/first_half/DataFrames/spatial_firing.pkl'):
        first_half_spatial_firing = pd.read_pickle(path + '/first_half/DataFrames/spatial_firing.pkl')
    else:
        return None, None, None, None
    if os.path.exists(path + '/second_half/DataFrames/spatial_firing.pkl'):
        second_half_spatial_firing = pd.read_pickle(path + '/second_half/DataFrames/spatial_firing.pkl')
    else:
        return None, None, None, None

    if os.path.exists(path + '/first_half/DataFrames/position.pkl'):
        first_position = pd.read_pickle(path + '/first_half/DataFrames/position.pkl')
    else:
        return None, None, None, None
    if os.path.exists(path + '/second_half/DataFrames/position.pkl'):
        second_position = pd.read_pickle(path + '/second_half/DataFrames/position.pkl')
    else:
        return None, None, None, None
    return first_half_spatial_firing, second_half_spatial_firing, first_position, second_position


def add_more_columns_to_cell_df(cell):
    cell['position_x_pixels'] = [np.array(cell.position_x_spikes.iloc[0]) * prm.get_pixel_ratio() / 100]
    cell['position_y_pixels'] = [np.array(cell.position_y_spikes.iloc[0]) * prm.get_pixel_ratio() / 100]
    cell['position_x'] = [np.array(cell.position_x_spikes.iloc[0])]
    cell['position_y'] = [np.array(cell.position_y_spikes.iloc[0])]
    cell['hd'] = [np.array(cell.hd_in_field_spikes.iloc[0])]
    cell['firing_times'] = [np.array(cell.spike_times.iloc[0])]
    return cell


def get_data_for_first_half(cell, spike_data_cluster_first, synced_spatial_data_first_half):
    first = pd.DataFrame()
    first['session_id'] = [cell.session_id.iloc[0]]
    first['cluster_id'] = [cell.cluster_id.iloc[0]]
    first['number_of_spikes'] = [len(spike_data_cluster_first.firing_times)]
    first['number_of_spikes_in_field'] = [len(spike_data_cluster_first.firing_times)]
    first['firing_times'] = [spike_data_cluster_first.firing_times]
    first['position_x'] = [spike_data_cluster_first.position_x]
    first['position_y'] = [spike_data_cluster_first.position_y]
    first['position_x_pixels'] = [spike_data_cluster_first.position_x_pixels]
    first['position_y_pixels'] = [spike_data_cluster_first.position_y_pixels]
    first['hd'] = [spike_data_cluster_first.hd]
    first['hd_in_field_spikes'] = [spike_data_cluster_first.hd]

    first['trajectory_x'] = [synced_spatial_data_first_half.position_x]
    first['trajectory_y'] = [synced_spatial_data_first_half.position_y]
    first['trajectory_hd'] = [synced_spatial_data_first_half.hd]
    first['hd_in_field_session'] = [synced_spatial_data_first_half.hd]
    first['trajectory_times'] = [synced_spatial_data_first_half.synced_time]
    first['time_spent_in_field'] = [len(synced_spatial_data_first_half.synced_time)]
    return first


def get_data_for_second_half(cell, spike_data_cluster_second, synced_spatial_data_second_half):
    second = pd.DataFrame()
    second['session_id'] = [cell.session_id.iloc[0]]
    second['cluster_id'] = [cell.cluster_id.iloc[0]]
    second['number_of_spikes'] = [len(spike_data_cluster_second.firing_times)]
    second['number_of_spikes_in_field'] = [len(spike_data_cluster_second.firing_times)]
    second['firing_times'] = [spike_data_cluster_second.firing_times]
    second['position_x'] = [spike_data_cluster_second.position_x]
    second['position_y'] = [spike_data_cluster_second.position_y]
    second['position_x_pixels'] = [spike_data_cluster_second.position_x_pixels]
    second['position_y_pixels'] = [spike_data_cluster_second.position_y_pixels]
    second['hd'] = [spike_data_cluster_second.hd]
    second['hd_in_field_spikes'] = [spike_data_cluster_second.hd]

    second['trajectory_x'] = [synced_spatial_data_second_half.position_x.reset_index(drop=True)]
    second['trajectory_y'] = [synced_spatial_data_second_half.position_y.reset_index(drop=True)]
    second['trajectory_hd'] = [synced_spatial_data_second_half.hd.reset_index(drop=True)]
    second['hd_in_field_session'] = [synced_spatial_data_second_half.hd.reset_index(drop=True)]
    second['trajectory_times'] = [synced_spatial_data_second_half.synced_time.reset_index(drop=True)]
    second['time_spent_in_field'] = [len(synced_spatial_data_second_half.synced_time.reset_index(drop=True))]
    return second


def add_hd_histogram_of_observed_data_to_df(fields):
    angles_session = np.array(fields.trajectory_hd[0])
    hd_hist_session = PostSorting.open_field_head_direction.get_hd_histogram(angles_session)
    hd_hist_session /= prm.get_sampling_rate()
    angles_spike = fields.hd[0]
    hd_hist_spikes = PostSorting.open_field_head_direction.get_hd_histogram(angles_spike)
    fields['hd_histogram_real_data_hz'] = [hd_hist_spikes / hd_hist_session / 1000]
    return fields


def split_in_two(cell):
    cell = add_more_columns_to_cell_df(cell)
    spike_data_in = cell
    synced_spatial_data_in = pd.DataFrame()
    synced_spatial_data_in['position_x'] = cell.position_x_session.iloc[0]
    synced_spatial_data_in['position_y'] = cell.position_y_session.iloc[0]
    synced_spatial_data_in['synced_time'] = cell.times_session.iloc[0]
    synced_spatial_data_in['hd'] = cell.hd_in_field_session.iloc[0]
    spike_data_in.set_index([spike_data_in.cluster_id - 1], inplace=True)
    spike_data_cluster_first, synced_spatial_data_first_half = PostSorting.compare_first_and_second_half.get_half_of_the_data_cell(prm, spike_data_in, synced_spatial_data_in, half='first_half')
    spike_data_cluster_second, synced_spatial_data_second_half = PostSorting.compare_first_and_second_half.get_half_of_the_data_cell(prm, spike_data_in, synced_spatial_data_in, half='second_half')

    synced_spatial_data_first_half['position_x_pixels'] = np.array(synced_spatial_data_first_half.position_x) * prm.get_pixel_ratio() / 100
    synced_spatial_data_first_half['position_y_pixels'] = np.array(synced_spatial_data_first_half.position_y) * prm.get_pixel_ratio() / 100
    synced_spatial_data_second_half['position_x_pixels'] = np.array(synced_spatial_data_second_half.position_x) * prm.get_pixel_ratio() / 100
    synced_spatial_data_second_half['position_y_pixels'] = np.array(synced_spatial_data_second_half.position_y) * prm.get_pixel_ratio() / 100

    first = get_data_for_first_half(cell, spike_data_cluster_first, synced_spatial_data_first_half)
    second = get_data_for_second_half(cell, spike_data_cluster_second, synced_spatial_data_second_half)

    first = add_hd_histogram_of_observed_data_to_df(first)
    second = add_hd_histogram_of_observed_data_to_df(second)

    return first, second, synced_spatial_data_first_half, synced_spatial_data_second_half


def plot_observed_vs_shuffled_correlations(observed, shuffled, cell):
    hd_polar_fig = plt.figure()
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    plt.hist(shuffled.flatten(), color='gray', alpha=0.8)
    ax.axvline(observed, color='navy')
    plt.xlim(-1, 1)
    plt.xticks([-1, 0, 1])
    plt.xlabel('Pearson correlation coef', fontsize=20)
    plt.ylabel('Number of shuffled cells', fontsize=20)
    plt.tight_layout()
    plt.savefig(local_path + cell.session_id[0] + str(cell.cluster_id[0]) + '_corr_coefs.png')
    plt.close()


def print_summary_stats(tag, corr_coefs_mean, percentiles):
    print('***********' + tag + '***************')
    print('avg corr correlations ')
    print(corr_coefs_mean)
    print('mean:')
    print(np.mean(corr_coefs_mean))
    print('std:')
    print(np.std(corr_coefs_mean))

    print('percentiles')
    print(percentiles)
    print('mean percentiles: ' + str(np.mean(percentiles)))
    print('sd percentiles: ' + str(np.std(percentiles)))
    print('number of cells in 95 percentile: ' + str(len(np.where(np.array(percentiles) > 95)[0])))
    print('number of all grid cells: ' + str(len(percentiles)))


def make_summary_plots(grid_data, percentiles):
    plt.cla()
    plt.scatter(grid_data.hd_score, percentiles, color='navy')
    plt.xlabel('Head direction score', fontsize=20)
    plt.ylabel('Percentile of correlation coef.', fontsize=20)
    plt.tight_layout()
    plt.savefig(local_path + 'pearson_coef_percentile_vs_hd_score.png')
    plt.close()

    plt.cla()
    plt.scatter(grid_data.number_of_spikes, percentiles, color='navy')
    plt.xlabel('Number of spikes', fontsize=20)
    plt.ylabel('Percentile of correlation coef.', fontsize=20)
    plt.tight_layout()
    plt.savefig(local_path + 'pearson_coef_percentile_vs_number_of_spikes.png')
    plt.close()


def get_half_rate_map_from_whole_cell(spatial_firing_all, session_id, cluster_id):
    cell = spatial_firing_all[(spatial_firing_all.session_id == session_id[0]) & (spatial_firing_all.cluster_id == cluster_id[0])]
    first_half, second_half, position_first, position_second = OverallAnalysis.compare_shuffled_from_first_and_second_halves.split_in_two(cell)
    position_heat_map_first, first_half = OverallAnalysis.open_field_firing_maps_processed_data.make_firing_field_maps(position_first, first_half, prm)
    position_heat_map_second, second_half = OverallAnalysis.open_field_firing_maps_processed_data.make_firing_field_maps(position_second, second_half, prm)
    return first_half, second_half, position_first, position_second


def add_rate_map_values_to_field(spatial_firing, field):
    spike_data_field = pd.DataFrame()
    spike_data_field['x'] = field.trajectory_x[0]
    spike_data_field['y'] = field.trajectory_y[0]
    spike_data_field['hd'] = field.trajectory_hd[0]
    spike_data_field['synced_time'] = field.trajectory_times[0]

    bin_size_pixels = PostSorting.open_field_firing_maps.get_bin_size(prm)
    spike_data_field['rate_map_x'] = (field.trajectory_x[0] // bin_size_pixels).astype(int)
    spike_data_field['rate_map_y'] = (field.trajectory_y[0] // bin_size_pixels).astype(int)
    rates = []
    cluster = spatial_firing.cluster_id == field.cluster_id
    rate_map = spatial_firing.firing_maps[cluster].iloc[0]
    for sample in range(len(spike_data_field)):
        rate = rate_map[spike_data_field.rate_map_x.iloc[sample], spike_data_field.rate_map_y.iloc[sample]]
        rates.append(rate)

    field['rate_map_values_session'] = [np.round(rates, 2)]
    return field


def distributive_shuffle(field_data, number_of_bins=20, number_of_times_to_shuffle=1000):
    field_histograms_all = []
    shuffled_hd_all = []
    for index, field in field_data.iterrows():
        print('I will shuffle data in the fields.')
        field_histograms = np.zeros((number_of_times_to_shuffle, number_of_bins))
        shuffle_indices = OverallAnalysis.shuffle_field_analysis.get_random_indices_for_shuffle(field, number_of_times_to_shuffle, shuffle_type='distributive')
        shuffled_hd_field = []
        for shuffle in range(number_of_times_to_shuffle):
            shuffled_hd = field['hd_in_field_session'][shuffle_indices[shuffle]]
            shuffled_hd_field.extend(shuffled_hd)
            hist, bin_edges = np.histogram(shuffled_hd, bins=number_of_bins, range=(0, 6.28))  # from 0 to 2pi
            field_histograms[shuffle, :] = hist
        field_histograms_all.append(field_histograms)
        shuffled_hd_all.append(shuffled_hd_field)
    field_data['shuffled_data'] = field_histograms_all
    field_data['shuffled_hd_distribution'] = shuffled_hd_all
    return field_data


def process_data(server_path, spike_sorter='/MountainSort', df_path='/DataFrames', sampling_rate_video=30, tag='mouse'):
    if tag == 'mouse':
        accepted_fields = pd.read_excel(local_path + 'list_of_accepted_fields.xlsx')
        all_cells = pd.read_pickle(OverallAnalysis.folder_path_settings.get_local_path() + '/compare_first_and_second_shuffled/all_mice_df.pkl')
    else:  # rat data
        accepted_fields = pd.read_excel(local_path + 'included_fields_detector2_sargolini.xlsx')
        all_cells = pd.read_pickle(OverallAnalysis.folder_path_settings.get_local_path() + '/compare_first_and_second_shuffled/all_rats_df.pkl')


    all_data = pd.read_pickle(local_path + 'all_' + tag + '_df.pkl')
    all_data = add_cell_types_to_data_frame(all_data)
    if tag == 'mouse':
        all_data = OverallAnalysis.shuffle_field_analysis_all_animals.tag_accepted_fields_mouse(all_data, accepted_fields)
    elif tag == 'rat':
        all_data = OverallAnalysis.shuffle_field_analysis_all_animals.tag_accepted_fields_rat(all_data, accepted_fields)
    grid_cells = all_data['cell type'] == 'grid'
    accepted = all_data['accepted_field'] == True
    grid_data = all_data[grid_cells & accepted]

    corr_coefs_mean = []
    corr_stds = []
    percentiles = []
    for iterator in range(len(grid_data)):
        print(iterator)
        print(grid_data.iloc[iterator].session_id)
        first_half, second_half, position_first, position_second = split_in_two(grid_data.iloc[iterator:iterator + 1])
        first_half_whole_cell, second_half_whole_cell, position_first_whole_cell, position_second_whole_cell = get_half_rate_map_from_whole_cell(all_cells, first_half.session_id, first_half.cluster_id)
        first_half = add_rate_map_values_to_field(first_half_whole_cell, first_half)
        first_half = distributive_shuffle(first_half, number_of_bins=20, number_of_times_to_shuffle=1000)
        first_half = OverallAnalysis.shuffle_field_analysis.analyze_shuffled_data(first_half, local_path + '/first/', sampling_rate_video, number_of_bins=20, shuffle_type='distributive')

        second_half = add_rate_map_values_to_field(second_half_whole_cell, second_half)
        second_half = distributive_shuffle(second_half, number_of_bins=20, number_of_times_to_shuffle=1000)
        second_half = OverallAnalysis.shuffle_field_analysis.analyze_shuffled_data(second_half, local_path + '/second/',
                                                                                  sampling_rate_video,
                                                                                  number_of_bins=20,
                                                                                  shuffle_type='distributive')

        # OverallAnalysis.shuffle_cell_analysis.plot_distributions_for_shuffled_vs_real_cells(spatial_firing_second, 'grid', animal=tag + str(iterator) + 'second', shuffle_type='distributive')

        print('shuffled')
        # compare
        # todo get time_spent_in_bins added to df somehow
        time_spent_in_bins_first = first_half.time_spent_in_bins  # based on trajectory
        # normalize shuffled data
        shuffled_histograms_hz_first = first_half.shuffled_data * sampling_rate_video / time_spent_in_bins_first
        time_spent_in_bins_second = second_half.time_spent_in_bins   # based on trajectory
        # normalize shuffled data
        shuffled_histograms_hz_second = second_half.shuffled_data * sampling_rate_video / time_spent_in_bins_second

        # look at correlations between rows of the two arrays above to get a distr of correlations for the shuffled data
        corr = np.corrcoef(shuffled_histograms_hz_first[0], shuffled_histograms_hz_second[0])[1000:, :1000]
        corr_mean = corr.mean()
        corr_std = corr.std()
        # check what percentile real value is relative to distribution of shuffled correlations
        first_half_hd_hist_hz, second_half_hd_hist_hz = array_utility.remove_nans_and_inf_from_both_arrays(first_half.hd_histogram_real_data_hz[0], second_half.hd_histogram_real_data_hz[0])
        corr_observed = scipy.stats.pearsonr(first_half_hd_hist_hz, second_half_hd_hist_hz)[0]

        plot_observed_vs_shuffled_correlations(corr_observed, corr, first_half)

        percentile = scipy.stats.percentileofscore(corr.flatten(), corr_observed)
        percentiles.append(percentile)

        corr_coefs_mean.append(corr_mean)
        corr_stds.append(corr_std)

    print_summary_stats(tag, corr_coefs_mean, percentiles)
    make_summary_plots(grid_data, percentiles)


def main():
    prm.set_pixel_ratio(440)
    prm.set_sampling_rate(30000)
    process_data(server_path_mouse, tag='mouse')
    prm.set_pixel_ratio(100)
    prm.set_sampling_rate(1)  # firing times are in seconds for rat data
    process_data(server_path_rat, tag='rat')


if __name__ == '__main__':
    main()