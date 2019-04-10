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


def plot_modes_python(real_cell, estimated_density, angles_x, angles_y):
    hd_polar_fig = plt.figure()
    hd_polar_fig.set_size_inches(5, 5, forward=True)
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    theta_estimate = np.linspace(0, 2 * np.pi, 3601)  # x axis
    theta_real = np.linspace(0, 2 * np.pi, 361)  # x axis
    ax = plt.subplot(1, 1, 1, polar=True)
    ax = plot_utility.style_polar_plot(ax)
    for mode in range(len(angles_x)):
        ax.plot([0, angles_x[mode]], [0, angles_y[mode]], zorder=3, color='navy')
    ax.plot(theta_estimate[:-1], list(estimated_density), color='black', linewidth=2)
    ax.plot(theta_real[:-1], list(real_cell), color='red', linewidth=2)
    plt.tight_layout()
    plt.show()
    plt.close()


def analyze_histograms(field_data):
    robj.r.source('count_modes_circular_histogram.R')
    mode_angles_x = []
    mode_angles_y = []
    fitted_densities = []
    for index, field in field_data.iterrows():
        hd_histogram_field = field.normalized_hd_hist
        # print(hd_histogram_field)
        if np.isnan(hd_histogram_field).sum() > 0:
            print('skipping this field, it has nans')
            fitted_density = np.nan
            angles_x = np.nan
            angles_y = np.nan
        else:
            print('I will analyze ' + field.session_id)
            resampled_distribution = resample_histogram(hd_histogram_field)
            fit = fit_von_mises_mixed_model(resampled_distribution)
            theta = get_model_fit_theta_value(fit)
            angles_x = theta[:int(len(theta) / 2)]
            angles_y = theta[int(len(theta) / 2):]
            fitted_density = get_estimated_density_function(fit)
            plot_modes_in_r(fit)
            plot_modes_python(hd_histogram_field, fitted_density, angles_x, angles_y)
        mode_angles_x.append(angles_x)
        mode_angles_y.append(angles_y)
        fitted_densities.append(fitted_density)
    field_data['fitted_density'] = fitted_densities
    field_data['mode_angles_x'] = mode_angles_x  # does not seem ok, max is very high
    field_data['mode_angles_y'] = mode_angles_y
    return field_data


def process_circular_data():
    print('I am loading the data frame that has the fields')
    field_data = pd.read_pickle(local_path + 'field_data_modes.pkl')
    field_data = analyze_histograms(field_data)


def main():
    process_circular_data()


if __name__ == '__main__':
    main()