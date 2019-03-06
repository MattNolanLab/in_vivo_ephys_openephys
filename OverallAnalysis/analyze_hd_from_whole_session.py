import glob
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import OverallAnalysis.false_positives
import OverallAnalysis.analyze_field_correlations
import os
import PostSorting.open_field_grid_cells

local_path = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/watson_two_test_cells/all_mice_df_2.pkl'
path_to_data = 'C://Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/watson_two_test_cells/'
save_output_path = 'C:/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/watson_two_test_cells/'
server_path = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/Open_field_opto_tagging_p038/'
local_output_path = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/watson_two_test_cells/all_mice_df_2.pkl'


def load_data_frame_spatial_firing(output_path):
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + '/MountainSort/DataFrames/spatial_firing.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            '''
            'session_id' 'cluster_id' 'tetrode' 'primary_channel' 'firing_times'
             'firing_times_opto' 'number_of_spikes' 'mean_firing_rate' 'isolation'
             'noise_overlap' 'peak_snr' 'peak_amp' 'random_snippets' 'position_x'
             'position_x_pixels' 'position_y' 'position_y_pixels' 'hd' 'speed'
             'hd_spike_histogram' 'max_firing_rate_hd' 'preferred_HD' 'hd_score'
             'firing_maps' 'max_firing_rate' 'firing_fields' 'field_max_firing_rate'
             'firing_fields_hd_session' 'firing_fields_hd_cluster' 'field_hd_max_rate'
             'field_preferred_hd' 'field_hd_score' 'number_of_spikes_in_fields'
             'time_spent_in_fields_sampling_points' 'spike_times_in_fields'
             'times_in_session_fields' 'field_corr_r' 'field_corr_p'
             'hd_correlation_first_vs_second_half'
             'hd_correlation_first_vs_second_half_p' 'hd_hist_first_half'
             'hd_hist_second_half'

            '''
            if ('hd_hist_first_half' in spatial_firing) and ('watson_test_hd' in spatial_firing):
                spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'number_of_spikes', 'mean_firing_rate', 'isolation', 'noise_overlap', 'peak_snr', 'hd_correlation_first_vs_second_half', 'hd_correlation_first_vs_second_half_p', 'hd_hist_first_half', 'firing_fields_hd_session', 'hd_hist_second_half', 'watson_test_hd', 'hd_score', 'hd', 'kuiper_cluster', 'watson_cluster', 'firing_maps']].copy()

                # print(spatial_firing.head())
                spatial_firing_data = spatial_firing_data.append(spatial_firing)

                print(spatial_firing_data.head())
    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def load_data_frame(path):
    if os.path.exists(local_path):
        df = pd.read_pickle(path)
    else:
        df = load_data_frame_spatial_firing(local_output_path)
    return df


def load_data_and_tag_false_positive_cells():
    false_positives_path = path_to_data + 'false_positives_all.txt'
    df_all_mice = load_data_frame(local_path)
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(false_positives_path)
    df_all_mice = add_combined_id_to_df(df_all_mice)
    df_all_mice['false_positive'] = df_all_mice['false_positive_id'].isin(list_of_false_positives)
    return df_all_mice


def plot_hd_vs_watson_stat(df_all_mice, save_output_path):
    good_cluster = df_all_mice.false_positive == False
    fig, ax = plt.subplots()
    hd_score = df_all_mice[good_cluster].hd_score
    watson_two_stat = df_all_mice[good_cluster].watson_test_hd
    plt.scatter(hd_score, watson_two_stat)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Head-direction score')
    ax.set_ylabel('Two-sample Watson test stat')
    plt.axhline(0.385, color='red')
    plt.savefig(save_output_path + 'hd_vs_watson_stat_all_cells.png')


def add_combined_id_to_df(df_all_mice):
    animal_ids = [session_id.split('_')[0] for session_id in df_all_mice.session_id.values]
    dates = [session_id.split('_')[1] for session_id in df_all_mice.session_id.values]
    tetrode = df_all_mice.tetrode.values
    cluster = df_all_mice.cluster_id.values

    combined_ids = []
    for cell in range(len(df_all_mice)):
        id = animal_ids[cell] + '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    df_all_mice['false_positive_id'] = combined_ids
    return df_all_mice


def correlation_between_first_and_second_halves_of_session(df_all_mice, save_output_path):
    excitatory_neurons = df_all_mice.mean_firing_rate <= 10
    inhibitory_neurons = df_all_mice.mean_firing_rate > 10
    good_cluster = df_all_mice.false_positive == False
    significant_corr = df_all_mice.hd_correlation_first_vs_second_half_p < 0.001
    watson_result_exists = df_all_mice.watson_test_hd.notnull()
    is_hd_cell = df_all_mice.hd_score >= 0.5
    print('Number of cells included in two sample watson test for head-direction from the whole session: ' + str(
        watson_result_exists.sum()))
    print('excitatory: ' + str(len(df_all_mice[excitatory_neurons & watson_result_exists])))
    print('inhibitory: ' + str(len(df_all_mice[inhibitory_neurons & watson_result_exists])))
    watson_significant = df_all_mice.watson_test_hd > 0.385  # p < 0.001
    print('Number of cells with significantly different HD distributions: ' + str(watson_significant.sum()))
    print('Number of excitatory neurons with significantly different HD: ' + str(len(df_all_mice[watson_significant & excitatory_neurons])))
    print('Number of inhibitory neurons with significantly different HD: ' + str(len(df_all_mice[watson_significant & inhibitory_neurons])))

    print('Number of excitatory neurons with significantly different HD that are hd cells: ' + str(len(df_all_mice[watson_significant & excitatory_neurons & is_hd_cell])))
    print('Number of inhibitory neurons with significantly different HD that are hd cells: ' + str(len(df_all_mice[watson_significant & inhibitory_neurons & is_hd_cell])))

    print('mean pearson r of correlation between first and second half')
    print(df_all_mice.hd_correlation_first_vs_second_half[significant_corr & good_cluster].mean())

    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_mice.hd_correlation_first_vs_second_half[significant_corr & good_cluster & watson_significant], save_output_path + 'correlation_hd_session.png')
    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_mice.hd_correlation_first_vs_second_half[significant_corr & good_cluster & watson_significant & excitatory_neurons], save_output_path + 'correlation_hd_session_excitatory.png')
    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_mice.hd_correlation_first_vs_second_half[significant_corr & good_cluster & watson_significant & inhibitory_neurons], save_output_path + 'correlation_hd_session_inhibitory.png')


def add_grid_score_to_df(df_all_mice):
    if 'grid_score' not in df_all_mice.columns:
        df_all_mice = PostSorting.open_field_grid_cells.process_grid_data(df_all_mice)
        df_all_mice.to_pickle(local_path)
    return df_all_mice


def plot_results_of_watson_test(df_all_mice, cell_type='grid'):
    good_cluster = df_all_mice.false_positive == False
    if cell_type == 'grid':
        cells_to_analyze = df_all_mice.grid_score >= 0.4
    elif cell_type == 'hd':
        cells_to_analyze = df_all_mice.hd_score >= 0.5
    else:
        not_grid = df_all_mice.grid_score < 0.4
        not_hd = df_all_mice.hd_score < 0.5
        cells_to_analyze = not_grid & not_hd

    watson_test_stats = df_all_mice.watson_cluster[good_cluster & cells_to_analyze]
    fig, ax = plt.subplots()
    plt.hist(watson_test_stats, bins=30, color='navy', normed=True)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.axvline(x=0.268, linewidth=5, color='red')  # p < 0.01 based on r docs for watson two test
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Watson test statistic', size=30)
    ax.set_ylabel('Proportion', size=30)
    plt.ylim(0, 0.05)
    plt.xlim(0, 700)
    ax.set_aspect(6000)
    plt.yticks([0, 0.05])
    plt.savefig(save_output_path + 'two_sample_watson_stats_hist_all_spikes_' + cell_type + '_cells.png', bbox_inches="tight")


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    df_all_mice = load_data_and_tag_false_positive_cells()
    correlation_between_first_and_second_halves_of_session(df_all_mice, save_output_path)
    plot_hd_vs_watson_stat(df_all_mice, save_output_path)
    add_grid_score_to_df(df_all_mice)
    plot_results_of_watson_test(df_all_mice, cell_type='grid')
    plot_results_of_watson_test(df_all_mice, cell_type='hd')
    plot_results_of_watson_test(df_all_mice, cell_type='nc')



if __name__ == '__main__':
    main()
