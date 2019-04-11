import cmath
import matplotlib.pylab as plt
import math_utility
import numpy as np
import pandas as pd
import plot_utility
from rpy2 import robjects as robj
from rpy2.robjects import pandas2ri


local_path = '/Users/s1466507/Dropbox/Edinburgh/grid_fields/analysis/field_modes/'


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


def plot_modes_python(real_cell, estimated_density, theta, field_id, path):
    hd_polar_fig = plt.figure()
    hd_polar_fig.set_size_inches(5, 5, forward=True)
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    theta_estimate = np.linspace(0, 2 * np.pi, 3601)  # x axis
    theta_real = np.linspace(0, 2 * np.pi, 361)  # x axis
    ax = plt.subplot(1, 1, 1, polar=True)
    ax = plot_utility.style_polar_plot(ax)
    lengthes = []
    angles = []
    number_of_modes = int(len(theta)/2)
    for mode in range(number_of_modes):
        length, angle = math_utility.cart2pol(np.asanyarray(theta)[mode][0], np.asanyarray(theta)[mode][1])
        lengthes.append(length)
        angles.append(angle)
    scale_for_lines = max(real_cell) / max(lengthes)

    for mode in range(number_of_modes):
        ax.plot((0, angles[mode]), (0, lengthes[mode]*scale_for_lines), color='red', linewidth=3)
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


def analyze_histograms(field_data):
    robj.r.source('count_modes_circular_histogram.R')
    mode_angles = []
    fitted_densities = []
    for index, field in field_data.iterrows():
        hd_histogram_field = field.normalized_hd_hist
        # print(hd_histogram_field)
        if np.isnan(hd_histogram_field).sum() > 0:
            print('skipping this field, it has nans')
            fitted_density = np.nan
            angles = np.nan
        else:
            print('I will analyze ' + field.session_id)
            resampled_distribution = resample_histogram(hd_histogram_field)
            fit = fit_von_mises_mixed_model(resampled_distribution)
            # alpha = get_model_fit_alpha_value(fit)  # probability, the relative strength of belief in that mode
            theta = get_model_fit_theta_value(fit)
            angles = get_mode_angles_degrees(theta)
            fitted_density = get_estimated_density_function(fit)
            if len(angles) > 0:
                pass
                # plot_modes_in_r(fit)
                # concentration = np.asanyarray(theta)[0]
                path = local_path + 'estimated_modes/' + field.session_id + str(field.cluster_id) + str(field.field_id)
                plot_modes_python(hd_histogram_field, fitted_density, theta, field.field_id, path)
        mode_angles.append(angles)
        fitted_densities.append(fitted_density)
    field_data['fitted_density'] = fitted_densities
    field_data['mode_angles'] = mode_angles  # does not seem ok, max is very high
    return field_data


def process_circular_data():
    print('I am loading the data frame that has the fields')
    field_data = pd.read_pickle(local_path + 'field_data_modes.pkl')
    field_data = analyze_histograms(field_data)


def main():
    process_circular_data()


if __name__ == '__main__':
    main()