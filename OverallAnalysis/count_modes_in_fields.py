import cmath
import glob
import matplotlib.pylab as plt
import math_utility
import numpy as np
import os
import pandas as pd
import plot_utility
from rpy2 import robjects as robj
from rpy2.robjects import pandas2ri


local_path = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/field_modes/'
server_path_mouse = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/Open_field_opto_tagging_p038/'
server_path_rat = '//ardbeg.mvm.ed.ac.uk/nolanlab/Klara/grid_field_analysis/moser_data/Sargolini/all_data/'


def load_field_data(output_path, server_path):
    if os.path.exists(output_path):
        field_data = pd.read_pickle(output_path)
        return field_data
    field_data_combined = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + '/MountainSort/DataFrames/shuffled_fields.pkl'
        if os.path.exists(data_frame_path):
            print('I found a field data frame.')
            field_data = pd.read_pickle(data_frame_path)
            if 'shuffled_data' in field_data:
                field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'indices_rate_map',
                                                    'spike_times', 'number_of_spikes_in_field', 'position_x_spikes',
                                                    'position_y_spikes', 'hd_in_field_spikes', 'hd_hist_spikes',
                                                    'times_session', 'time_spent_in_field', 'position_x_session',
                                                    'position_y_session', 'hd_in_field_session', 'hd_hist_session',
                                                    'hd_histogram_real_data', 'time_spent_in_bins',
                                                    'field_histograms_hz']].copy()
                field_data_to_combine['normalized_hd_hist'] = field_data.hd_hist_spikes / field_data.hd_hist_session

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
def add_cell_types_to_data_frame_rat(field_data):
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


# todo: replace this with python implementation
def read_cell_type_from_accepted_clusters(field_data, accepted_fields):
    accepted_fields_to_merge = accepted_fields[['unique_id', 'cell type', 'grid score', 'hd score']]
    field_data_merged = pd.merge(field_data, accepted_fields_to_merge, on='unique_id')
    return field_data_merged


def resample_histogram(histogram):
    number_of_times_to_sample = robj.r(1000)
    hd_cluster_r = robj.FloatVector(histogram)
    rejection_sampling_r = robj.r['rejection.sampling']
    resampled_distribution = rejection_sampling_r(number_of_times_to_sample, hd_cluster_r)
    return resampled_distribution


def fit_von_mises_mixed_model(resampled_distribution):
    fit_von_mises_mix = robj.r('vMFmixture')
    find_best_fit = robj.r('vMFmin')
    fit = find_best_fit(fit_von_mises_mix(resampled_distribution))
    print(fit)
    return fit


def get_model_fit_alpha_value(fit):
    get_model_fit_alpha = robj.r('get_model_fit_alpha')
    alpha = get_model_fit_alpha(fit)
    return alpha


def get_model_fit_theta_value(fit):
    get_model_fit_theta = robj.r('get_model_fit_theta')
    theta = get_model_fit_theta(fit)
    return theta


def get_estimated_density_function(fit):
    get_estimated_density = robj.r('get_estimated_density')
    estimated_density = get_estimated_density(fit)
    return estimated_density


def plot_modes_in_r(fit):
    plot_modes = robj.r("plot_modes")
    plot_modes(fit)


def generate_colors(number_of_firing_fields):
    colors = [[0, 1, 0], [1, 0.6, 0.3], [0, 1, 1], [1, 0, 1], [0.7, 0.3, 1], [0.6, 0.5, 0.4], [0.6, 0, 0]]  # green, orange, cyan, pink, purple, grey, dark red
    if number_of_firing_fields > len(colors):
        for i in range(number_of_firing_fields):
            colors.append(plot_utility.generate_new_color(colors, pastel_factor=0.9))
    return colors


def find_angles_and_lengths(theta):
    lengths = []
    angles = []
    number_of_modes = int(len(theta)/2)
    for mode in range(number_of_modes):
        length, angle = math_utility.cart2pol(np.asanyarray(theta)[mode][0], np.asanyarray(theta)[mode][1])
        lengths.append(length)
        angles.append(angle)
    return angles, lengths


def plot_modes_python(real_cell, estimated_density, theta, field_id, path):
    hd_polar_fig = plt.figure()
    hd_polar_fig.set_size_inches(5, 5, forward=True)
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    theta_estimate = np.linspace(0, 2 * np.pi, 3601)  # x axis
    theta_real = np.linspace(0, 2 * np.pi, 361)  # x axis
    ax = plt.subplot(1, 1, 1, polar=True)
    ax = plot_utility.style_polar_plot(ax)
    angles, lengths = find_angles_and_lengths(theta)
    scale_for_lines = max(real_cell) / max(lengths)
    number_of_modes = int(len(theta) / 2)
    for mode in range(number_of_modes):
        ax.plot((0, angles[mode]), (0, lengths[mode]*scale_for_lines), color='red', linewidth=3)
    scale_for_density = max(real_cell) / max(estimated_density)
    ax.plot(theta_estimate[:-1], estimated_density*scale_for_density, color='black', linewidth=2)
    colors = generate_colors(field_id + 1)
    ax.plot(theta_real[:-1], list(real_cell), color=colors[field_id], linewidth=2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_mode_angles_degrees(theta):
    angles = []
    for mode in range(int(len(theta)/2)):
        length, angle = math_utility.cart2pol(np.asanyarray(theta)[mode][0], np.asanyarray(theta)[mode][1])
        angle *= 180 / np.pi
        angles.append(angle)
    return angles


def plot_modes_in_field(field, hd_histogram_field, fitted_density, theta):
    # plot_modes_in_r(fit)
    # concentration = np.asanyarray(theta)[0]
    path = local_path + 'estimated_modes/' + field.session_id + str(field.cluster_id) + str(field.field_id)
    plot_modes_python(hd_histogram_field, fitted_density, theta, field.field_id, path)


def analyze_histograms(field_data):
    robj.r.source('count_modes_circular_histogram.R')
    mode_angles = []
    angles_stds = []
    fitted_densities = []
    for index, field in field_data.iterrows():
        hd_histogram_field = field.normalized_hd_hist
        if np.isnan(hd_histogram_field).sum() > 0:
            print('skipping this field, it has nans')
            fitted_density = np.nan
            angles = np.nan
            angles_std = np.nan
        else:
            print('I will analyze ' + field.session_id)
            resampled_distribution = resample_histogram(hd_histogram_field)
            fit = fit_von_mises_mixed_model(resampled_distribution)
            # alpha = get_model_fit_alpha_value(fit)  # probability, the relative strength of belief in that mode
            theta = get_model_fit_theta_value(fit)
            angles = get_mode_angles_degrees(theta)
            fitted_density = get_estimated_density_function(fit)
            angles_std = np.nan
            if len(angles) > 0:
                plot_modes_in_field(field, hd_histogram_field, fitted_density, theta)
                angles_std = np.std(angles)
        mode_angles.append(angles)
        fitted_densities.append(fitted_density)
        angles_stds.append(angles_std)
    field_data['fitted_density'] = fitted_densities
    field_data['mode_angles'] = mode_angles  # does not seem ok, max is very high
    field_data['angles_std'] = angles_stds
    return field_data


def plot_std_of_modes(field_data, animal):
    print(animal + ' modes are analyzed')
    grid_cells = field_data['cell type'] == 'grid'
    conjunctive_cells = field_data['cell type'] == 'conjunctive'
    accepted_field = field_data['accepted_field'] == True
    print('cell')
    grid_modes_std = field_data[accepted_field & grid_cells].angles_std
    conjunctive_modes_std = field_data[accepted_field & conjunctive_cells].angles_std
    plt.hist(grid_modes_std[~np.isnan(grid_modes_std)], color='navy')
    plt.hist(conjunctive_modes_std[~np.isnan(conjunctive_modes_std)], color='red')
    plt.show()
    plt.close()
    plt.savefig(local_path + animal + '_std_of_modes_of_grid_and_conj_cells.png')


def process_circular_data(animal):
    print('I am loading the data frame that has the fields')
    if animal == 'mouse':
        field_data = load_field_data(local_path + 'field_data_modes_mouse.pkl', server_path_mouse)
        field_data = analyze_histograms(field_data)
        accepted_fields = pd.read_excel(local_path + 'list_of_accepted_fields.xlsx')
        field_data = tag_accepted_fields_mouse(field_data, accepted_fields)
        # field_data = read_cell_type_from_accepted_clusters(field_data, accepted_fields)
        field_data = read_cell_type_from_accepted_clusters(field_data, accepted_fields)

    if animal == 'rat':
        accepted_fields = pd.read_excel(local_path + 'included_fields_detector2_sargolini.xlsx')
        field_data = tag_accepted_fields_rat(field_data, accepted_fields)
        field_data = add_cell_types_to_data_frame_rat(field_data)
    plot_std_of_modes(field_data, 'mouse')




def main():
    process_circular_data('mouse')
    process_circular_data('rat')


if __name__ == '__main__':
    main()