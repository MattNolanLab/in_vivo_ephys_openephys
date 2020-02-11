import data_frame_utility
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_field_analysis
import OverallAnalysis.compare_shuffled_from_first_and_second_halves_fields
import pandas as pd
import PostSorting.parameters

import scipy
import matplotlib.pylab as plt


local_path = OverallAnalysis.folder_path_settings.get_local_path()
analysis_path = local_path + '/super_long_recording/'

prm = PostSorting.parameters.Parameters()
prm.set_pixel_ratio(440)
prm.set_sampling_rate(30000)


def get_shuffled_field_data(spatial_firing, position_data, shuffle_type='distributive', sampling_rate_video=50):
    field_df = data_frame_utility.get_field_data_frame(spatial_firing, position_data)
    field_df = OverallAnalysis.shuffle_field_analysis.add_rate_map_values_to_field_df_session(spatial_firing, field_df)
    field_df = OverallAnalysis.shuffle_field_analysis.shuffle_field_data(field_df, analysis_path, number_of_bins=20,
                                  number_of_times_to_shuffle=1000, shuffle_type=shuffle_type)
    field_df = OverallAnalysis.shuffle_field_analysis.analyze_shuffled_data(field_df, analysis_path, sampling_rate_video,
                                     number_of_bins=20, shuffle_type=shuffle_type)
    return field_df


def get_number_of_directional_fields(fields, tag='grid'):
    percentiles_no_correction = []
    percentiles_correction = []
    for index, field in fields.iterrows():
        percentile = scipy.stats.percentileofscore(field.number_of_different_bins_shuffled, field.number_of_different_bins)
        percentiles_no_correction.append(percentile)

        percentile = scipy.stats.percentileofscore(field.number_of_different_bins_shuffled_corrected_p, field.number_of_different_bins_bh)
        percentiles_correction.append(percentile)

    print(tag)
    print('Number of fields: ' + str(len(fields)))
    print('Number of directional fields [without correction]: ')
    print(np.sum(np.array(percentiles_no_correction) > 95))
    fields['directional_no_correction'] = np.array(percentiles_no_correction) > 95

    print('Number of directional fields [with BH correction]: ')
    print(np.sum(np.array(percentiles_correction) > 95))
    fields['directional_correction'] = np.array(percentiles_correction) > 95
    print('Percentile values, with correction:')
    print(percentiles_correction)


def add_trajectory_data_to_spatial_firing_df(position, spatial_firing):
    spatial_firing['trajectory_x'] = [position.position_x] * len(spatial_firing)
    spatial_firing['trajectory_y'] = [position.position_y] * len(spatial_firing)
    spatial_firing['trajectory_hd'] = [position.hd] * len(spatial_firing)
    spatial_firing['trajectory_times'] = [position.synced_time] * len(spatial_firing)
    return spatial_firing


def plot_shuffled_number_of_bins_vs_observed(cell):
    # percentile = scipy.stats.percentileofscore(cell.number_of_different_bins_shuffled_corrected_p.iloc[0], cell.number_of_different_bins_bh.iloc[0])
    for index, field in cell.iterrows():
        shuffled_distribution = cell.number_of_different_bins_shuffled_corrected_p.iloc[index]
        plt.cla()
        fig = plt.figure(figsize=(6, 3))
        plt.yticks([0, 500, 1000])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yticklabels([0, '', 1000])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.hist(shuffled_distribution, bins=range(20), color='gray')
        ax.axvline(x=cell.number_of_different_bins_bh.iloc[0], color='navy', linewidth=3)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        # plt.xscale('log')
        max_x_value = max(cell.number_of_different_bins_bh.iloc[0], shuffled_distribution.max())
        plt.xlim(0, max_x_value + 1)
        plt.ylabel('N (shuffles)', fontsize=24)
        plt.xlabel('N (significant bins)', fontsize=24)
        plt.tight_layout()
        plt.savefig(analysis_path + str(index) + 'number_of_significant_bars_shuffled_vs_real_example.png')
        plt.close()


def process_data():
    firing = pd.read_pickle(analysis_path + 'DataFrames/spatial_firing.pkl')
    position = pd.read_pickle(analysis_path + 'DataFrames/position.pkl')
    # load shuffled field data
    if os.path.exists(analysis_path + 'DataFrames/fields.pkl'):
        shuffled_fields = pd.read_pickle(analysis_path + 'DataFrames/fields.pkl')
    else:
        firing = pd.read_pickle(analysis_path + 'DataFrames/spatial_firing.pkl')
        position = pd.read_pickle(analysis_path + 'DataFrames/position.pkl')
        shuffled_fields = get_shuffled_field_data(firing, position)
        shuffled_fields.to_pickle(analysis_path + 'DataFrames/fields.pkl')

    firing = add_trajectory_data_to_spatial_firing_df(position, firing)

    number_of_significant_bins = shuffled_fields.number_of_different_bins_bh
    print(number_of_significant_bins)
    get_number_of_directional_fields(shuffled_fields, tag='grid')
    col_names = ['session_id', 'cluster_id', 'field_id', 'corr_coefs_mean', 'shuffled_corr_median', 'corr_stds', 'percentiles', 'hd_scores_all',
                 'number_of_spikes_all', 'spatial_scores', 'percentages_of_excluded_bins', 'spatial_scores_field',
                 'percentages_of_excluded_bins_field', 'unsampled_hds']
    aggregated_data_field_correlations = pd.DataFrame(columns=col_names)
    plot_shuffled_number_of_bins_vs_observed(shuffled_fields)

    sampling_rate_video = 30
    for iterator in range(len(shuffled_fields)):
        aggregated_data_field_correlations = OverallAnalysis.compare_shuffled_from_first_and_second_halves_fields.compare_observed_and_shuffled_correlations(iterator, shuffled_fields, firing, aggregated_data_field_correlations, sampling_rate_video)


def main():
    process_data()


if __name__ == '__main__':
    main()
