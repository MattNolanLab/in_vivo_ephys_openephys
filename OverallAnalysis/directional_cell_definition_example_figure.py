import numpy as np
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_field_analysis_heading
import pandas as pd
import plot_utility
import PostSorting.open_field_heading_direction
import PostSorting.open_field_firing_maps
import matplotlib.pylab as plt
import scipy.stats
import os
import glob

local_path = OverallAnalysis.folder_path_settings.get_local_path()
analysis_path = local_path + '/methods_directional_field/'


def plot_shuffled_number_of_bins_vs_observed(cell):
    percentile = scipy.stats.percentileofscore(cell.number_of_different_bins_shuffled_corrected_p.iloc[0], cell.number_of_different_bins_bh.iloc[0])
    shuffled_distribution = cell.number_of_different_bins_shuffled_corrected_p.iloc[0]
    plt.figure()
    plt.hist(shuffled_distribution, color='gray')
    plt.axvline(x=percentile, color='blue')
    plt.xscale('log')
    plt.ylabel('Number of shuffles', fontsize=24)
    plt.xlabel('Number of significant bins (log)', fontsize=24)
    plt.savefig(analysis_path + 'number_of_significant_bars_shuffled_vs_real_example.png')
    plt.close()


def get_number_of_directional_cells(cells, tag='grid'):
    print('HEAD DIRECTION')
    percentiles_no_correction = []
    percentiles_correction = []
    for index, cell in cells.iterrows():
        percentile = scipy.stats.percentileofscore(cell.number_of_different_bins_shuffled, cell.number_of_different_bins)
        percentiles_no_correction.append(percentile)

        percentile = scipy.stats.percentileofscore(cell.number_of_different_bins_shuffled_corrected_p, cell.number_of_different_bins_bh)
        percentiles_correction.append(percentile)

    print(tag)
    print('Number of fields: ' + str(len(cells)))
    print('Number of directional cells [without correction]: ')
    print(np.sum(np.array(percentiles_no_correction) > 95))
    cells['directional_no_correction'] = np.array(percentiles_no_correction) > 95

    print('Number of directional cells [with BH correction]: ')
    print(np.sum(np.array(percentiles_correction) > 95))
    cells['directional_correction'] = np.array(percentiles_correction) > 95


def make_example_plot():
    session_id = 'M12_2018-04-10_14-22-14_of'
    # load shuffled hd data
    spatial_firing = pd.read_pickle(analysis_path + 'all_mice_df.pkl')
    '''
    ['session_id', 'cluster_id', 'hd_score', 'position_x', 'position_y',
       'hd', 'firing_maps', 'number_of_spikes_in_fields', 'firing_times',
       'trajectory_hd', 'trajectory_x', 'trajectory_y', 'trajectory_times',
       'number_of_spikes', 'rate_map_autocorrelogram', 'grid_spacing',
       'field_size', 'grid_score', 'false_positive_id', 'false_positive',
       'shuffled_data', 'shuffled_means', 'shuffled_std',
       'hd_histogram_real_data_hz', 'time_spent_in_bins',
       'shuffled_histograms_hz', 'shuffled_percentile_threshold_95',
       'shuffled_percentile_threshold_5', 'error_bar_95', 'error_bar_5',
       'real_and_shuffled_data_differ_bin', 'number_of_different_bins',
       'number_of_different_bins_shuffled', 'percentile_of_observed_data',
       'shuffle_p_values', 'p_values_corrected_bars_bh',
       'p_values_corrected_bars_holm', 'number_of_different_bins_bh',
       'number_of_different_bins_holm',
       'number_of_different_bins_shuffled_corrected_p']
    
    '''
    example_session = spatial_firing.session_id == session_id
    example_cell = spatial_firing[example_session]
    get_number_of_directional_cells(example_cell, tag='grid')
    plot_shuffled_number_of_bins_vs_observed(example_cell)
    # make plots of example shuffles
    # make overall distribution plot
    pass


def main():
    make_example_plot()


if __name__ == '__main__':
    main()