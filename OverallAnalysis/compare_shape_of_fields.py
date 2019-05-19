import glob
import matplotlib.pylab as plt
import math_utility
import math
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import pandas as pd
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots
import plot_utility
from rpy2 import robjects as robj
from scipy.stats import circstd
from rpy2.robjects import pandas2ri
import scipy.stats
import seaborn
import PostSorting.compare_first_and_second_half
import PostSorting.parameters

prm = PostSorting.parameters.Parameters()
prm.set_sorter_name('MountainSort')


local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/field_histogram_shapes/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()


def load_field_data(output_path, server_path, spike_sorter):
    if os.path.exists(output_path):
        field_data = pd.read_pickle(output_path)
        return field_data
    field_data_combined = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + '/DataFrames/shuffled_fields.pkl'
        spatial_firing_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a field data frame.')
            field_data = pd.read_pickle(data_frame_path)
            spatial_firing = pd.read_pickle(spatial_firing_path)
            position = pd.read_pickle(position_path)
            prm.set_file_path(recording_folder)
            # spatial_firing = PostSorting.compare_first_and_second_half.analyse_first_and_second_halves(prm, position, spatial_firing)
            if 'shuffled_data' in field_data:
                field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'position_x_spikes',
                                                    'position_y_spikes', 'position_x_session', 'position_y_session',
                                                    'field_histograms_hz', 'indices_rate_map', 'hd_in_field_spikes',
                                                    'hd_in_field_session', 'spike_times', 'times_session']].copy()
                field_data_to_combine['normalized_hd_hist'] = field_data.hd_hist_spikes / field_data.hd_hist_session
                if 'hd_score' in field_data:
                    field_data_to_combine['hd_score'] = field_data.hd_score
                if 'grid_score' in field_data:
                    field_data_to_combine['grid_score'] = field_data.grid_score
                rate_maps = []
                length_recording = []
                for cluster in range(len(field_data.cluster_id)):
                    rate_map = spatial_firing[field_data.cluster_id.iloc[cluster] == spatial_firing.cluster_id].firing_maps
                    rate_maps.append(rate_map)
                    length_of_recording = (position.synced_time.max() - position.synced_time.min())
                    length_recording.append(length_of_recording)

                field_data_to_combine['rate_map'] = rate_maps
                field_data_to_combine['recording_length'] = length_recording
                add_histograms_for_half_recordings(field_data_to_combine, )
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
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


# add cell type tp rat data frame
def add_cell_types_to_data_frame(field_data):
    cell_type = []
    for index, field in field_data.iterrows():
        if field.hd_score >= 0.5 and field.grid_score >= 0.4:
            cell_type.append('conjunctive')
        elif field.hd_score >= 0.5:
            cell_type.append('hd')
        elif field.grid_score >= 0.4:
            cell_type.append('grid')
        else:
            cell_type.append('na')

    field_data['cell type'] = cell_type

    return field_data


def clean_data(coefs):
    flat_coefs = [item for sublist in coefs for item in sublist]
    return [x for x in flat_coefs if str(x) != 'nan']


def format_bar_chart(ax, x_label, y_label):
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(x_label, fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    return ax


def plot_pearson_coefs_of_field_hist(coefs_grid, coefs_conjunctive, animal, tag=''):
    grid_coefs = clean_data(coefs_grid)
    conj_coefs = clean_data(coefs_conjunctive)
    fig, ax = plt.subplots()
    ax = format_bar_chart(ax, 'Pearson correlation coef.', 'Proportion')
    plt.hist(grid_coefs, color='navy', alpha=0.7, normed=True)
    if len(conj_coefs) > 0:
        plt.hist(conj_coefs, color='red', alpha=0.7, normed='True')
    plt.savefig(local_path + animal + tag + '_correlation_of_field_histograms.png')
    plt.close()


def remove_nans(field1, field2):
    not_nans_in_field1 = ~np.isnan(field1)
    not_nans_in_field2 = ~np.isnan(field2)
    field1 = field1[not_nans_in_field1 & not_nans_in_field2]
    field2 = field2[not_nans_in_field1 & not_nans_in_field2]
    return field1, field2


def compare_hd_histograms(field_data):
    field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)
    list_of_cells = np.unique(list(field_data.unique_cell_id))
    pearson_coefs_all = []
    pearson_coefs_avg = []
    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        field_histograms = field_data.loc[field_data['unique_cell_id'] == cell_id].normalized_hd_hist
        pearson_coefs_cell = []
        for index1, field1 in enumerate(field_histograms):
            for index2, field2 in enumerate(field_histograms):
                if index1 != index2:
                    field1_clean, field2 = remove_nans(field1, field2)
                    pearson_coef = scipy.stats.pearsonr(field1_clean, field2)[0]
                    pearson_coefs_cell.append(pearson_coef)
        pearson_coefs_all.append([pearson_coefs_cell])
        pearson_coefs_avg.append([np.mean(pearson_coefs_cell)])
    return pearson_coefs_avg


def save_correlation_plot(corr, animal, cell_type, tag=''):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    plt.cla()
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    # cmap = seaborn.color_palette("RdBu_r", 10)
    cmap = seaborn.color_palette("coolwarm", 10)

    # Draw the heatmap with the mask and correct aspect ratio
    seaborn.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(local_path + animal + '_' + cell_type + tag + '_correlation_heatmap.png')
    plt.close()


def get_field_df_to_correlate(histograms):
    histograms_df = pd.DataFrame()
    field_number = 0
    for index, cell in histograms.iterrows():
        field_number += 1
        histograms_df[str(field_number)] = cell.normalized_hd_hist
    return histograms_df


def get_field_df_to_correlate_halves(histograms):
    histograms_df = pd.DataFrame()
    field_number = 0
    for index, cell in histograms.iterrows():
        field_number += 1
        histograms_df[str(field_number)] = cell.hd_hist_first_half

    for index, cell in histograms.iterrows():
        field_number += 1
        histograms_df[str(field_number + int(len(histograms)/2))] = cell.hd_hist_second_half
    return histograms_df


def plot_correlation_matrix(field_data, animal):
    grid_histograms = field_data.loc[(field_data.grid_score >= 0.4) & (field_data.hd_score < 0.5) & (field_data.accepted_field == True)]
    grid_histograms_df = get_field_df_to_correlate(grid_histograms)
    # Compute the correlation matrix
    corr = grid_histograms_df.corr()
    save_correlation_plot(corr, animal, 'grid')

    grid_histograms_centre = field_data.loc[(field_data.grid_score >= 0.4) & (field_data.hd_score < 0.5) & (field_data.accepted_field == True) & (field_data.border_field == False)]
    grid_histograms_df_centre = get_field_df_to_correlate(grid_histograms_centre)
    # Compute the correlation matrix
    corr = grid_histograms_df_centre.corr()
    save_correlation_plot(corr, animal, 'grid', 'centre_fields')

    grid_histograms_border = field_data.loc[(field_data.grid_score >= 0.4) & (field_data.hd_score < 0.5) & (field_data.accepted_field == True) & (field_data.border_field == True)]
    grid_histograms_df_border = get_field_df_to_correlate(grid_histograms_border)
    # Compute the correlation matrix
    corr = grid_histograms_df_border.corr()
    save_correlation_plot(corr, animal, 'grid', 'border_fields')

    conjunctive_histograms = field_data.loc[(field_data.grid_score >= 0.4) & (field_data.hd_score >= 0.5) & (field_data.accepted_field == True)]
    conjunctive_histograms_df = get_field_df_to_correlate(conjunctive_histograms)
    # Compute the correlation matrix
    corr = conjunctive_histograms_df.corr()
    save_correlation_plot(corr, animal, 'conjunctive')

    grid_histograms = field_data.loc[(field_data.grid_score >= 0.4) & (field_data.hd_score < 0.5) & (field_data.accepted_field == True)]
    grid_histograms_df_halves = get_field_df_to_correlate_halves(grid_histograms)
    # Compute the correlation matrix
    corr = grid_histograms_df_halves.corr()
    save_correlation_plot(corr, animal, 'grid_halves')


def plot_correlation_matrix_individual_cells(field_data, animal):
    field_data = field_data[field_data.accepted_field == True]
    field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)
    list_of_cells = np.unique(list(field_data.unique_cell_id))
    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        field_histograms = field_data.loc[field_data['unique_cell_id'] == cell_id]
        field_df = get_field_df_to_correlate(field_histograms)
        corr = field_df.corr()  # correlation matrix
        save_correlation_plot(corr, animal, '', tag=cell_id)

    field_data_centre = field_data[(field_data.accepted_field == True) & (field_data.border_field == False)]
    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        field_histograms = field_data_centre.loc[field_data_centre['unique_cell_id'] == cell_id]
        field_df = get_field_df_to_correlate(field_histograms)
        corr = field_df.corr()  # correlation matrix
        save_correlation_plot(corr, animal, '', tag=cell_id + '_centre')

    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        field_histograms = field_data_centre.loc[field_data_centre['unique_cell_id'] == cell_id]
        field_df = get_field_df_to_correlate_halves(field_histograms)
        corr = field_df.corr()  # correlation matrix
        save_correlation_plot(corr, animal, '', tag=cell_id + '_halves')


# if it touches the border it's a border field
def tag_border_and_middle_fields(field_data):
    border_tag = []
    for index, field in field_data.iterrows():
        rate_map = field.rate_map.iloc[0]
        field_indices = field.indices_rate_map
        y_max = len(rate_map)
        x_max = len(rate_map[0])
        border = False
        if (field_indices[:, 0] < 1).sum() + (field_indices[:, 0] == x_max).sum() > 0:
            border = True
        if (field_indices[:, 1] < 1).sum() + (field_indices[:, 1] == y_max).sum() > 0:
            border = True
        border_tag.append(border)

    field_data['border_field'] = border_tag
    return field_data


def add_histograms_for_half_recordings(field_data, sampling_rate_ephys, sampling_rate_movement, output_path):
    #if 'hd_hist_first_half' in field_data:
        #return field_data
    first_halves = []
    second_halves = []
    pearson_coefs = []
    pearson_ps = []
    number_of_spikes_first_half = []
    number_of_spikes_second_half = []
    time_spent_first_half = []
    time_spent_second_half = []
    for field_index, field in field_data.iterrows():
        # half_time_sec = (max(field.times_session) - min(field.times_session)) / 2
        half_time = field.recording_length / 2
        session_first_half_indices = np.array(field.times_session) < half_time
        session_first_half = field.hd_in_field_session[session_first_half_indices]
        hd_hist_first_half_session = PostSorting.open_field_head_direction.get_hd_histogram(session_first_half)
        session_second_half_indices = np.array(field.times_session) >= half_time
        session_second_half = field.hd_in_field_session[session_second_half_indices]
        hd_hist_second_half_session = PostSorting.open_field_head_direction.get_hd_histogram(session_second_half)
        spikes_first_half = np.array(field.spike_times) < (half_time * sampling_rate_ephys)
        spikes_second_half = np.array(field.spike_times) >= (half_time * sampling_rate_ephys)
        hd_first_half_spikes = field.hd_in_field_spikes[spikes_first_half]
        hd_second_half_spikes = field.hd_in_field_spikes[spikes_second_half]
        hd_hist_first_half = PostSorting.open_field_head_direction.get_hd_histogram(hd_first_half_spikes)
        hd_hist_second_half = PostSorting.open_field_head_direction.get_hd_histogram(hd_second_half_spikes)
        hd_hist_norm_first = np.divide(hd_hist_first_half, hd_hist_first_half_session, out=np.zeros_like(hd_hist_first_half), where=hd_hist_first_half_session != 0)
        first_halves.append(hd_hist_norm_first)
        hd_hist_norm_second = np.divide(hd_hist_second_half, hd_hist_second_half_session, out=np.zeros_like(hd_hist_second_half), where=hd_hist_second_half_session != 0)
        second_halves.append(hd_hist_norm_second)
        pearson_coef = scipy.stats.pearsonr(hd_hist_norm_first, hd_hist_norm_second)[0]
        pearson_p = scipy.stats.pearsonr(hd_hist_norm_first, hd_hist_norm_second)[1]
        pearson_coefs.append(pearson_coef)
        pearson_ps.append(pearson_p)
        number_of_spikes_first_half.append(spikes_first_half.sum())
        number_of_spikes_second_half.append(spikes_second_half.sum())
        time_spent_first_half.append(session_first_half_indices.sum())
        time_spent_second_half.append(session_second_half_indices.sum())

    field_data['hd_hist_first_half'] = first_halves
    field_data['hd_hist_second_half'] = second_halves
    field_data['hd_in_first_and_second_halves_corr'] = pearson_coefs
    field_data['hd_in_first_and_second_halves_p'] = pearson_ps
    field_data['number_of_spikes_first_half'] = number_of_spikes_first_half
    field_data['number_of_spikes_second_half'] = number_of_spikes_second_half
    field_data['time_spent_first_half'] = time_spent_first_half
    field_data['time_spent_second_half'] = time_spent_second_half

    field_data.to_pickle(output_path)
    return field_data


def compare_within_field_with_other_fields(field_data, animal):
    first_halves = field_data.hd_hist_first_half
    second_halves = field_data.hd_hist_second_half
    correlation_values = []
    correlation_p = []
    count_f1 = 0
    count_f2 = 0
    for index, field1 in enumerate(first_halves):
        for index2, field2 in enumerate(second_halves):
            if count_f1 != count_f2:
                pearson_coef, corr_p = scipy.stats.pearsonr(field1, field2)
                correlation_values.append(pearson_coef)
                correlation_p.append(corr_p)
            count_f2 += 1
        count_f1 += 1

    correlation_values_in_between = np.array(correlation_values)
    correlation_p = np.array(correlation_p)
    # significant = field_data.hd_in_first_and_second_halves_p < 0.001
    within_field = field_data.hd_in_first_and_second_halves_corr
    fig, ax = plt.subplots()
    ax = format_bar_chart(ax, 'Pearson correlation coef.', 'Proportion')
    # in_between_fields = correlation_values_in_between[correlation_p < 0.001]
    in_between_fields = correlation_values_in_between
    plt.hist(in_between_fields[~np.isnan(in_between_fields)], weights=plot_utility.get_weights_normalized_hist(in_between_fields[~np.isnan(in_between_fields)]), color='gray', alpha=0.5)
    plt.hist(within_field[~np.isnan(within_field)], weights=plot_utility.get_weights_normalized_hist(within_field[~np.isnan(within_field)]), color='blue', alpha=0.4)
    plt.xlim(-1, 1)
    plt.savefig(local_path + animal + 'half_session_correlations.png')
    plt.close()


def compare_within_field_with_other_fields_correlating_fields(field_data, animal):
    first_halves = field_data.hd_hist_first_half.values
    second_halves = field_data.hd_hist_second_half.values
    correlation_within = field_data.hd_in_first_and_second_halves_corr.values
    correlation_values = []
    correlation_p = []
    count_f1 = 0
    count_f2 = 0

    for index, field1 in enumerate(first_halves):
        for index2, field2 in enumerate(second_halves):
            if count_f1 != count_f2:
                if (correlation_within[index] >= 0.4) & (correlation_within[index2] >= 0.5):
                    pearson_coef, corr_p = scipy.stats.pearsonr(field1, field2)
                    correlation_values.append(pearson_coef)
                    correlation_p.append(corr_p)
            count_f2 += 1
        count_f1 += 1

    correlation_values_in_between = np.array(correlation_values)
    correlation_p = np.array(correlation_p)
    # significant = field_data.hd_in_first_and_second_halves_p < 0.001
    within_field = field_data.hd_in_first_and_second_halves_corr.values
    within_field = within_field[within_field >= 0.4]
    fig, ax = plt.subplots()
    ax = format_bar_chart(ax, 'Pearson correlation coef.', 'Proportion')
    # in_between_fields = correlation_values_in_between[correlation_p < 0.001]
    in_between_fields = correlation_values_in_between
    plt.hist(in_between_fields[~np.isnan(in_between_fields)], weights=plot_utility.get_weights_normalized_hist(in_between_fields[~np.isnan(in_between_fields)]), color='gray', alpha=0.5)
    plt.hist(within_field[~np.isnan(within_field)], weights=plot_utility.get_weights_normalized_hist(within_field[~np.isnan(within_field)]), color='blue', alpha=0.4)
    plt.xlim(-1, 1)
    plt.savefig(local_path + animal + 'half_session_correlations_internally_correlating_only_r04.png')
    plt.close()


def plot_half_fields(field_data, animal):
    for index, field in field_data.iterrows():
        plt.cla()
        fig, ax = plt.subplots()
        corr = str(round(field.hd_in_first_and_second_halves_corr, 4))
        p = str(round(field.hd_in_first_and_second_halves_p, 4))
        cell_type = field['cell type']
        number_of_spikes_first_half = field['number_of_spikes_first_half']
        number_of_spikes_second_half = field['number_of_spikes_second_half']
        time_spent_first_half = field['time_spent_first_half']
        time_spent_second_half = field['time_spent_second_half']

        # title = ('r= ' + corr + ' p=' + p + ' cell type: ' + cell_type + 'first half: ' + str(number_of_spikes_first_half) + ' ' + str(time_spent_first_half) + ' second half: ' + str(number_of_spikes_second_half) + ' ' + str(time_spent_second_half))
        title = ('r= ' + corr + ' p=' + p )
        save_path = local_path + '/first_vs_second_fields/' + animal + field.session_id + str(field.cluster_id) + str(field.field_id)
        PostSorting.open_field_make_plots.plot_polar_hd_hist(field.hd_hist_first_half, field.hd_hist_second_half, field.cluster_id, save_path, color1='lime', color2='navy', title=title)

        plt.close()


def process_circular_data(animal):
    # print('I am loading the data frame that has the fields')
    if animal == 'mouse':
        mouse_path = local_path + 'field_data_modes_mouse.pkl'
        field_data = load_field_data(mouse_path, server_path_mouse, '/MountainSort')
        accepted_fields = pd.read_excel(local_path + 'list_of_accepted_fields.xlsx')
        field_data = tag_accepted_fields_mouse(field_data, accepted_fields)
        field_data = add_cell_types_to_data_frame(field_data)
        field_data = add_histograms_for_half_recordings(field_data, 30000, 30, mouse_path)
        field_data = tag_border_and_middle_fields(field_data)
        grid_cell_pearson = compare_hd_histograms(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid')])
        grid_pearson_centre = compare_hd_histograms(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid') & (field_data.border_field == False)])
        conjunctive_cell_pearson = compare_hd_histograms(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'conjunctive')])
        conjunctive_pearson_centre = compare_hd_histograms(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'conjunctive') & (field_data.border_field == False)])

        compare_within_field_with_other_fields_correlating_fields(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid')], 'grid_mouse')
        compare_within_field_with_other_fields(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid')], 'grid_mouse')
        compare_within_field_with_other_fields(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'conjunctive')], 'conj_mouse')

        # plot_pearson_coefs_of_field_hist(grid_cell_pearson, conjunctive_cell_pearson, 'mouse')
        # plot_pearson_coefs_of_field_hist(grid_pearson_centre, conjunctive_pearson_centre, 'mouse', tag='_centre')
        # plot_correlation_matrix(field_data, 'mouse')
        # plot_correlation_matrix_individual_cells(field_data, 'mouse')
        # plot_half_fields(field_data, 'mouse')

    if animal == 'rat':
        rat_path = local_path + 'field_data_modes_rat.pkl'
        field_data = load_field_data(rat_path, server_path_rat, '')
        accepted_fields = pd.read_excel(local_path + 'included_fields_detector2_sargolini.xlsx')
        field_data = tag_accepted_fields_rat(field_data, accepted_fields)
        field_data = add_cell_types_to_data_frame(field_data)
        field_data = add_histograms_for_half_recordings(field_data, 1, 50, rat_path)  # firing times are in seconds
        field_data = tag_border_and_middle_fields(field_data)
        grid_cell_pearson = compare_hd_histograms(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid')])
        grid_pearson_centre = compare_hd_histograms(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid') & (field_data.border_field == False)])
        conjunctive_cell_pearson = compare_hd_histograms(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'conjunctive')])
        conjunctive_pearson_centre = compare_hd_histograms(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'conjunctive') & (field_data.border_field == False)])

        compare_within_field_with_other_fields_correlating_fields(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid')], 'grid_rat')
        compare_within_field_with_other_fields(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid')], 'rat')

        # plot_pearson_coefs_of_field_hist(grid_cell_pearson, conjunctive_cell_pearson, 'rat')
        # plot_pearson_coefs_of_field_hist(grid_pearson_centre, conjunctive_pearson_centre, 'rat', tag='_centre')
        # plot_correlation_matrix(field_data, 'rat')
        # plot_correlation_matrix_individual_cells(field_data, 'rat')
        # plot_half_fields(field_data, 'rat')


def main():
    process_circular_data('mouse')
    process_circular_data('rat')


if __name__ == '__main__':
    main()