import data_frame_utility
import matplotlib.pylab as plt
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_cell_analysis
import OverallAnalysis.compare_shuffled_from_first_and_second_halves_fields
import OverallAnalysis.false_positives
import pandas as pd
import PostSorting.parameters
import plot_utility

import scipy


local_path = OverallAnalysis.folder_path_settings.get_local_path()
analysis_path = local_path + '/plot_hd_tuning_vs_shuffled/'

prm = PostSorting.parameters.Parameters()
prm.set_pixel_ratio(440)
prm.set_sampling_rate(30000)


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
    spatial_firing['animal'] = animal_ids

    dates = [session_id.split('_')[1] for session_id in spatial_firing.session_id.values]

    cluster = spatial_firing.cluster_id.values
    combined_ids = []
    for cell in range(len(spatial_firing)):
        id = animal_ids[cell] + '-' + dates[cell] + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    spatial_firing['false_positive_id'] = combined_ids
    return spatial_firing


def tag_false_positives(spatial_firing):
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(analysis_path + 'false_positives_all.txt')
    spatial_firing = add_combined_id_to_df(spatial_firing)
    spatial_firing['false_positive'] = spatial_firing['false_positive_id'].isin(list_of_false_positives)
    return spatial_firing


def plot_bar_chart_for_cells_percentile_error_bar(spatial_firing, path, animal, shuffle_type='occupancy'):
    counter = 0
    for index, cell in spatial_firing.iterrows():
        mean = cell['shuffled_means']
        percentile_95 = cell['error_bar_95']
        percentile_5 = cell['error_bar_5']
        shuffled_histograms_hz = cell['shuffled_histograms_hz']
        hd_polar_fig = plt.figure()

        # x_pos = np.arange(shuffled_histograms_hz.shape[1])
        x_pos = np.linspace(0, 2*np.pi, shuffled_histograms_hz.shape[1])
        fig, ax = plt.subplots()
        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        # ax = OverallAnalysis.shuffle_cell_analysis.format_bar_chart(ax)
        # ax.errorbar(x_pos, mean, yerr=[percentile_5, percentile_95], alpha=0.7, color='black', ecolor='grey', capsize=10, fmt='o', markersize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        ax.plot(x_pos, cell.hd_histogram_real_data_hz, color='navy', linewidth=10)
        plt.scatter(x_pos, cell.hd_histogram_real_data_hz, marker='o', color='navy', s=40)
        plt.title('Number of spikes ' + str(cell.number_of_spikes))
        plt.savefig(analysis_path + animal + '_' + shuffle_type + '/' + str(counter) + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile_polar')
        plt.close()
        counter += 1


def plot_hd_vs_shuffled():
    mouse_df_path = analysis_path + 'all_mice_df.pkl'
    mouse_df = pd.read_pickle(mouse_df_path)
    df = tag_false_positives(mouse_df)
    good_cells = df.false_positive == False
    df_good_cells = df[good_cells]
    df = add_cell_types_to_data_frame(df_good_cells)
    grid_cells = df['cell type'] == 'grid'
    df_grid = df[grid_cells]
    print('mouse')
    plot_bar_chart_for_cells_percentile_error_bar(df_grid, analysis_path, 'mouse', shuffle_type='distributive')


def main():
    plot_hd_vs_shuffled()


if __name__ == '__main__':
    main()