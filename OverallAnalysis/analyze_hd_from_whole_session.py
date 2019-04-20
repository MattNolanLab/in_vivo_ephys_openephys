import glob
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import OverallAnalysis.false_positives
import OverallAnalysis.folder_path_settings
import OverallAnalysis.analyze_field_correlations
import os

import rpy2.robjects as ro
from rpy2.robjects.packages import importr


local_path_mouse = OverallAnalysis.folder_path_settings.get_local_path() + '/watson_two_test_cells/all_mice_df.pkl'
local_path_rat = OverallAnalysis.folder_path_settings.get_local_path() + '/watson_two_test_cells/all_rats_df.pkl'
path_to_data = OverallAnalysis.folder_path_settings.get_local_path() + '/watson_two_test_cells/'
save_output_path = OverallAnalysis.folder_path_settings.get_local_path() + '/watson_two_test_cells/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()


# run 2 sample watson test and put it in df
def run_two_sample_watson_test(hd_cluster, hd_session):
    circular = importr("circular")
    watson_two_test = circular.watson_two_test
    hd_cluster = ro.FloatVector(hd_cluster)
    hd_session = ro.FloatVector(hd_session)
    stat = watson_two_test(hd_cluster, hd_session)
    return stat[0][0]  # this is the part of the return r object that is the stat


# call R to tun two sample watson test on HD from firing field when the cell fired vs HD when the mouse was in the field
def compare_hd_when_the_cell_fired_to_heading(cell_data, position):
    two_watson_stats = []
    for index, cell in cell_data.iterrows():
        print('two-sample watson test on ' + str(cell.session_id) + str(cell.cluster_id))
        if type(cell.hd) == list:
            hd_cluster = (np.asanyarray(cell.hd) + 180) * np.pi / 180
        else:
            hd_cluster = (cell.hd + 180) * np.pi / 180
        hd_session = (position.hd + 180) * np.pi / 180
        two_watson_stat = run_two_sample_watson_test(hd_cluster, hd_session)
        two_watson_stats.append(two_watson_stat)
    cell_data['watson_test_hd'] = two_watson_stats
    return cell_data


def load_spatial_firing(output_path, server_path, animal, spike_sorter=''):
    if os.path.exists(output_path):
        spatial_firing = pd.read_pickle(output_path)
        return spatial_firing
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_data_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            position_data = pd.read_pickle(position_data_path)
            if 'grid_score' in spatial_firing:
                if animal == 'rat':
                    spatial_firing = spatial_firing[['session_id', 'cell_id', 'cluster_id', 'firing_times',
                                                    'number_of_spikes', 'hd', 'speed', 'mean_firing_rate',
                                                     'hd_spike_histogram', 'max_firing_rate_hd', 'preferred_HD',
                                                     'grid_spacing', 'field_size', 'grid_score', 'hd_score', 'firing_fields']].copy()
                if animal == 'mouse':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'firing_times',
                                                     'number_of_spikes', 'hd', 'speed', 'mean_firing_rate',
                                                     'hd_spike_histogram', 'max_firing_rate_hd', 'preferred_HD',
                                                     'grid_spacing', 'field_size', 'grid_score', 'hd_score',
                                                     'firing_fields']].copy()

                spatial_firing = compare_hd_when_the_cell_fired_to_heading(spatial_firing, position_data)
                spatial_firing_data = spatial_firing_data.append(spatial_firing)

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def load_data_and_tag_false_positive_cells():
    false_positives_path = path_to_data + 'false_positives_all.txt'
    df_all_mice = load_spatial_firing(local_path_mouse, server_path_mouse, 'mouse', spike_sorter='/MountainSort')
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(false_positives_path)
    df_all_mice = add_combined_id_to_df(df_all_mice)
    df_all_mice['false_positive'] = df_all_mice['false_positive_id'].isin(list_of_false_positives)
    return df_all_mice


def plot_hd_vs_watson_stat(df_all_cells, animal='mouse'):
    good_cluster = df_all_cells.false_positive == False
    grid_cell = (df_all_cells.grid_score >= 0.4) & (df_all_cells.hd_score < 0.5)
    fig, ax = plt.subplots()
    hd_score = df_all_cells[good_cluster].hd_score
    watson_two_stat = df_all_cells[good_cluster].watson_test_hd
    plt.scatter(hd_score, watson_two_stat)
    hd_score_grid = df_all_cells[good_cluster & grid_cell].hd_score
    watson_two_stat_grid = df_all_cells[good_cluster & grid_cell].watson_test_hd
    plt.scatter(hd_score_grid, watson_two_stat_grid, color='red')
    plt.xscale('log')
    plt.yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Head-direction score')
    ax.set_ylabel('Two-sample Watson test stat')
    plt.axhline(0.268, color='red')
    plt.savefig(save_output_path + 'hd_vs_watson_stat_all_cells_' + animal + '_log.png')


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


def correlation_between_first_and_second_halves_of_session(df_all_animals, animal='mouse'):
    excitatory_neurons = df_all_animals.mean_firing_rate <= 10
    inhibitory_neurons = df_all_animals.mean_firing_rate > 10
    good_cluster = df_all_animals.false_positive == False
    significant_corr = df_all_animals.hd_correlation_first_vs_second_half_p < 0.001
    watson_result_exists = df_all_animals.watson_test_hd.notnull()
    is_hd_cell = df_all_animals.hd_score >= 0.5
    print('Number of cells included in two sample watson test for head-direction from the whole session: ' + str(
        watson_result_exists.sum()))
    print('excitatory: ' + str(len(df_all_animals[excitatory_neurons & watson_result_exists])))
    print('inhibitory: ' + str(len(df_all_animals[inhibitory_neurons & watson_result_exists])))
    watson_significant = df_all_animals.watson_test_hd > 0.268  # p < 0.01
    print('Number of cells with significantly different HD distributions: ' + str(watson_significant.sum()))
    print('Number of excitatory neurons with significantly different HD: ' + str(len(df_all_animals[watson_significant & excitatory_neurons])))
    print('Number of inhibitory neurons with significantly different HD: ' + str(len(df_all_animals[watson_significant & inhibitory_neurons])))

    print('Number of excitatory neurons with significantly different HD that are hd cells: ' + str(len(df_all_animals[watson_significant & excitatory_neurons & is_hd_cell])))
    print('Number of inhibitory neurons with significantly different HD that are hd cells: ' + str(len(df_all_animals[watson_significant & inhibitory_neurons & is_hd_cell])))

    print('mean pearson r of correlation between first and second half')
    print(df_all_animals.hd_correlation_first_vs_second_half[significant_corr & good_cluster].mean())

    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_animals.hd_correlation_first_vs_second_half[significant_corr & good_cluster & watson_significant], save_output_path + 'correlation_hd_session_' + animal + '.png', y_axis_label='Number of cells')
    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_animals.hd_correlation_first_vs_second_half[significant_corr & good_cluster & watson_significant & excitatory_neurons], save_output_path + 'correlation_hd_session_excitatory_' + animal + '.png', y_axis_label='Number of cells')
    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_animals.hd_correlation_first_vs_second_half[significant_corr & good_cluster & watson_significant & inhibitory_neurons], save_output_path + 'correlation_hd_session_inhibitory_' + animal + '.png', y_axis_label='Number of cells')


def plot_results_of_watson_test(df_all_animals, cell_type='grid', animal='mouse'):
    good_cluster = df_all_animals.false_positive == False
    if cell_type == 'grid':
        grid = df_all_animals.grid_score >= 0.4
        not_hd = df_all_animals.hd_score < 0.5
        cells_to_analyze = grid & not_hd
    elif cell_type == 'hd':
        not_grid = df_all_animals.grid_score < 0.4
        hd = df_all_animals.hd_score >= 0.5
        cells_to_analyze = not_grid & hd
    else:
        not_grid = df_all_animals.grid_score < 0.4
        not_hd = df_all_animals.hd_score < 0.5
        cells_to_analyze = not_grid & not_hd

    watson_test_stats = df_all_animals.watson_test_hd[good_cluster & cells_to_analyze]
    watson_test_stats = watson_test_stats[~np.isnan(watson_test_stats)]

    print('\n' + animal)
    print(cell_type)
    print('all cells: ' + str(len(watson_test_stats)))
    print('significant: ' + str(len(watson_test_stats > 0.268)))

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
    plt.savefig(save_output_path + animal + '_two_sample_watson_stats_hist_all_spikes_' + cell_type + '_cells.png', bbox_inches="tight")


def process_mouse_data():
    print('-------------------------------------------------------------')
    df_all_mice = load_data_and_tag_false_positive_cells()
    # correlation_between_first_and_second_halves_of_session(df_all_mice)
    plot_hd_vs_watson_stat(df_all_mice)
    plot_results_of_watson_test(df_all_mice, cell_type='grid')
    plot_results_of_watson_test(df_all_mice, cell_type='hd')
    plot_results_of_watson_test(df_all_mice, cell_type='nc')


def process_rat_data():
    all_rats = load_spatial_firing(local_path_rat, server_path_rat, 'rat', spike_sorter='')
    all_rats['false_positive'] = np.full(len(all_rats), False)
    plot_hd_vs_watson_stat(all_rats, animal='rat')
    plot_results_of_watson_test(all_rats, cell_type='grid', animal='rat')
    plot_results_of_watson_test(all_rats, cell_type='hd', animal='rat')
    plot_results_of_watson_test(all_rats, cell_type='nc', animal='rat')
    # correlation_between_first_and_second_halves_of_session(all_rats, animal='rat')


def main():
    process_mouse_data()
    process_rat_data()


if __name__ == '__main__':
    main()
