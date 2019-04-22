import numpy as np
import OverallAnalysis.folder_path_settings
import pandas as pd
import matplotlib.pylab as plt
import scipy.stats
import os
import glob

analysis_path = OverallAnalysis.folder_path_settings.get_local_path() + '/shuffled_analysis/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()

local_path_to_shuffled_field_data_mice = analysis_path + 'shuffled_field_data_all_mice.pkl'
local_path_to_shuffled_field_data_rats = analysis_path + 'shuffled_field_data_all_rats.pkl'


# loads shuffle analysis results for field data
def load_data_frame_field_data(output_path, server_path, spike_sorter):
    if os.path.exists(output_path):
        field_data = pd.read_pickle(output_path)
        return field_data

    else:
        field_data_combined = pd.DataFrame()
        for recording_folder in glob.glob(server_path + '*'):
            os.path.isdir(recording_folder)
            data_frame_path = recording_folder + spike_sorter + '/DataFrames/shuffled_fields.pkl'
            if os.path.exists(data_frame_path):
                print('I found a field data frame.')
                field_data = pd.read_pickle(data_frame_path)
                if 'field_id' in field_data:
                    field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'indices_rate_map',
                                                        'spike_times', 'number_of_spikes_in_field', 'position_x_spikes',
                                                        'position_y_spikes', 'hd_in_field_spikes', 'hd_hist_spikes',
                                                        'times_session', 'time_spent_in_field', 'position_x_session',
                                                        'position_y_session', 'hd_in_field_session', 'hd_hist_session',
                                                        'hd_histogram_real_data', 'time_spent_in_bins',
                                                        'field_histograms_hz', 'hd_score', 'grid_score', 'shuffled_means', 'shuffled_std',
                                                        'real_and_shuffled_data_differ_bin', 'number_of_different_bins',
                                                        'number_of_different_bins_shuffled', 'number_of_different_bins_bh',
                                                        'number_of_different_bins_holm', 'number_of_different_bins_shuffled_corrected_p']].copy()

                    field_data_combined = field_data_combined.append(field_data_to_combine)
                    print(field_data_combined.head())
    field_data_combined.to_pickle(output_path)
    return field_data_combined


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields_mouse(field_data, accepted_fields):
    unique_id = field_data.session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)
    field_data['unique_id'] = unique_id
    unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    accepted_fields['unique_id'] = unique_id
    field_data['unique_cell_id'] = field_data.session_id + '_' + field_data.cluster_id.apply(str)
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields_rat(field_data, accepted_fields):
    unique_id = field_data.session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)
    unique_cell_id = field_data.session_id + '_' + field_data.cluster_id.apply(str)
    field_data['unique_id'] = unique_id
    field_data['unique_cell_id'] = unique_cell_id
    if 'Session ID' in accepted_fields:
        unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    else:
        unique_id = accepted_fields['SessionID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)

    accepted_fields['unique_id'] = unique_id
    field_data['unique_cell_id'] = field_data.session_id + '_' + field_data.cluster_id.apply(str)
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


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
    ax.set_xlim(0, 20)
    ax.set_xlabel('Rejected bars / field', size=30)
    ax.set_ylabel('Proportion', size=30)
    plt.savefig(analysis_path + 'distribution_of_rejects_' + animal + '.png', bbox_inches="tight")
    plt.close()


def plot_histogram_of_number_of_rejected_bars_shuffled(shuffled_field_data, animal='mouse'):
    number_of_rejects = shuffled_field_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for field in number_of_rejects:
        flat_shuffled.extend(field)
    fig, ax = plt.subplots()
    plt.hist(flat_shuffled, color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Rejected bars / field', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_xlim(0, 20)
    plt.savefig(analysis_path + '/distribution_of_rejects_shuffled' + animal + '.png', bbox_inches="tight")
    plt.close()


def make_combined_plot_of_distributions(shuffled_field_data, tag='grid'):
    tail, percentile_95, percentile_99 = find_tail_of_shuffled_distribution_of_rejects(shuffled_field_data)

    number_of_rejects_shuffled = shuffled_field_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for field in number_of_rejects_shuffled:
        flat_shuffled.extend(field)
    fig, ax = plt.subplots()
    plt.hist(flat_shuffled, normed=True, color='black', alpha=0.5)

    number_of_rejects_real = shuffled_field_data.number_of_different_bins
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
    ax.set_xlabel('Rejected bars / field', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_xlim(0, 20)
    plt.savefig(analysis_path + 'distribution_of_rejects_combined_all_' + tag + '.png', bbox_inches = "tight")
    plt.close()


def plot_number_of_significant_p_values(field_data, type='bh'):
    if type == 'bh':
        number_of_significant_p_values = field_data.number_of_different_bins_bh
    else:
        number_of_significant_p_values = field_data.number_of_different_bins_holm

    fig, ax = plt.subplots()
    plt.hist(number_of_significant_p_values, normed='True', color='navy', alpha=0.5)
    flat_shuffled = []
    for field in field_data.number_of_different_bins_shuffled_corrected_p:
        flat_shuffled.extend(field)
    plt.hist(flat_shuffled, normed='True', color='gray', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Rejected bars / field', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_ylim(0, 0.2)
    ax.set_xlim(0, 20)
    plt.savefig(analysis_path + 'distribution_of_rejects_significant_p_ ' + type + '.png', bbox_inches = "tight")
    plt.close()


def compare_distributions(x, y):
    stat, p = scipy.stats.mannwhitneyu(x, y)
    return p


def compare_shuffled_to_real_data_mw_test(field_data, analysis_type='bh'):
    if analysis_type == 'bh':
        flat_shuffled = []
        for field in field_data.number_of_different_bins_shuffled_corrected_p:
            flat_shuffled.extend(field)
            p_bh = compare_distributions(field_data.number_of_different_bins_bh, flat_shuffled)
            print('p value for comparing shuffled distribution to B-H corrected p values: ' + str(p_bh))
            return p_bh

    if analysis_type == 'percentile':
        flat_shuffled = []
        for field in field_data.number_of_different_bins_shuffled:
            flat_shuffled.extend(field)
            p_percentile = compare_distributions(field_data.number_of_different_bins, flat_shuffled)
            print('p value for comparing shuffled distribution to percentile thresholded p values: ' + str(p_percentile))
            return p_percentile


def plot_distributions_for_fields(shuffled_field_data, tag='grid', animal='mouse'):
    plot_histogram_of_number_of_rejected_bars(shuffled_field_data, animal)
    plot_histogram_of_number_of_rejected_bars_shuffled(shuffled_field_data, animal)
    plot_number_of_significant_p_values(shuffled_field_data, type='bh_' + tag + '_' + animal)
    plot_number_of_significant_p_values(shuffled_field_data, type='holm_' + tag + '_' + animal)
    make_combined_plot_of_distributions(shuffled_field_data, tag=tag + '_' + animal)


def analyze_data(animal):
    if animal == 'mouse':
        local_path_to_field_data = local_path_to_shuffled_field_data_mice
        server_path = server_path_mouse
        spike_sorter = '/MountainSort'
        accepted_fields = pd.read_excel(analysis_path + 'list_of_accepted_fields.xlsx')
    else:
        local_path_to_field_data = local_path_to_shuffled_field_data_rats
        server_path = server_path_rat
        spike_sorter = ''
        accepted_fields = pd.read_excel(analysis_path + 'included_fields_detector2_sargolini.xlsx')

    shuffled_field_data = load_data_frame_field_data(local_path_to_field_data, server_path, spike_sorter)
    if animal == 'mouse':
        tag_accepted_fields_mouse(shuffled_field_data, accepted_fields)
    else:
        shuffled_field_data = tag_accepted_fields_rat(shuffled_field_data, accepted_fields)
    grid = shuffled_field_data.grid_score >= 0.4
    hd = shuffled_field_data.hd_score >= 0.5
    not_classified = np.logical_and(np.logical_not(grid), np.logical_not(hd))
    grid_cells = np.logical_and(grid, np.logical_not(hd))

    shuffled_field_data_grid = shuffled_field_data[grid_cells]
    shuffled_field_data_not_classified = shuffled_field_data[not_classified]

    plot_distributions_for_fields(shuffled_field_data_grid, 'grid', animal=animal)
    plot_distributions_for_fields(shuffled_field_data_not_classified, 'not_classified', animal=animal)

    print(animal + ' data:')
    print('Grid cells:')
    print('Number of grid fields: ' + str(len(shuffled_field_data_grid)))
    print('Number of grid cells:' + str(len(np.unique(list(shuffled_field_data_grid.unique_cell_id)))))
    compare_shuffled_to_real_data_mw_test(shuffled_field_data_grid, analysis_type='bh')
    compare_shuffled_to_real_data_mw_test(shuffled_field_data_grid, analysis_type='percentile')
    print('__________________________________')
    print('Not classified cells:')
    print('Number of not classified fields: ' + str(len(shuffled_field_data_not_classified)))
    print('Number of not classified cells:' + str(len(np.unique(list(shuffled_field_data_not_classified.unique_cell_id)))))
    compare_shuffled_to_real_data_mw_test(shuffled_field_data_not_classified, analysis_type='bh')
    compare_shuffled_to_real_data_mw_test(shuffled_field_data_not_classified, analysis_type='percentile')
    print('__________________________________')


def main():
    analyze_data('mouse')
    analyze_data('rat')


if __name__ == '__main__':
    main()
