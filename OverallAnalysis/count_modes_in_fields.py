import matplotlib.pylab as plt
import numpy as np
import pandas as pd
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


def analyze_histograms(field_data):
    robj.r.source('count_modes_circular_histogram.R')
    for index, field in field_data.iterrows():
        hd_histogram_field = field.normalized_hd_hist
        # print(hd_histogram_field)
        if np.isnan(hd_histogram_field).sum() > 0:
            print('skipping this field, it has nans')
            resampled_distribution = np.nan
        else:
            print('I will analyze ' + field.session_id)
            resampled_distribution = resample_histogram(hd_histogram_field)
            fit = fit_von_mises_mixed_model(resampled_distribution)
            get_model_fit_alpha = robj.r('get_model_fit_alpha')
            get_model_fit_theta = robj.r('get_model_fit_theta')
            alpha = get_model_fit_alpha(fit)
            theta = get_model_fit_theta(fit)
            get_estimated_density = robj.r('get_estimated_density')
            estimated_density = get_estimated_density(fit)
            plot_modes =robj.r("plot_modes")
            # plot_modes(fit)
            print(theta)  # these are the estimated modes (polar coordinates)
    return field_data


def process_circular_data():
    print('I am loading the data frame that has the fields')
    field_data = pd.read_pickle(local_path + 'field_data_modes.pkl')
    field_data = analyze_histograms(field_data)


def main():
    process_circular_data()


if __name__ == '__main__':
    main()