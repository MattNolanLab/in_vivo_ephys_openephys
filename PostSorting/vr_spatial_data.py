import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
from scipy import stats
import PostSorting.vr_speed_analysis
import plot_utility
import settings
from astropy.convolution import convolve, Gaussian1DKernel


def calculate_total_trial_numbers(raw_position_data,processed_position_data):
    print('calculating total trial numbers for trial types')
    trial_numbers = np.array(raw_position_data['trial_number'])
    trial_type = np.array(raw_position_data['trial_type'])
    trial_data=np.transpose(np.vstack((trial_numbers, trial_type)))
    beaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]>0),0)
    unique_beaconed_trials = np.unique(beaconed_trials[:,0])
    nonbeaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]!=1),0)
    unique_nonbeaconed_trials = np.unique(nonbeaconed_trials[1:,0])
    probe_trials = np.delete(trial_data, np.where(trial_data[:,1]!=2),0)
    unique_probe_trials = np.unique(probe_trials[1:,0])

    processed_position_data.at[0,'beaconed_total_trial_number'] = len(unique_beaconed_trials)
    processed_position_data.at[0,'nonbeaconed_total_trial_number'] = len(unique_nonbeaconed_trials)
    processed_position_data.at[0,'probe_total_trial_number'] = len(unique_probe_trials)
    return processed_position_data

def trial_average_speed(processed_position_data):
    # split binned speed data by trial type
    beaconed = processed_position_data[processed_position_data["trial_type"] == 0]
    non_beaconed = processed_position_data[processed_position_data["trial_type"] == 1]
    probe = processed_position_data[processed_position_data["trial_type"] == 2]

    if len(beaconed)>0:
        beaconed_speeds = plot_utility.pandas_collumn_to_2d_numpy_array(beaconed["speeds_binned"])
        trial_averaged_beaconed_speeds = np.nanmean(beaconed_speeds, axis=0)
    else:
        trial_averaged_beaconed_speeds = np.array([])

    if len(non_beaconed)>0:
        non_beaconed_speeds = plot_utility.pandas_collumn_to_2d_numpy_array(non_beaconed["speeds_binned"])
        trial_averaged_non_beaconed_speeds = np.nanmean(non_beaconed_speeds, axis=0)
    else:
        trial_averaged_non_beaconed_speeds = np.array([])

    if len(probe)>0:
        probe_speeds = plot_utility.pandas_collumn_to_2d_numpy_array(probe["speeds_binned"])
        trial_averaged_probe_speeds = np.nanmean(probe_speeds, axis=0)
    else:
        trial_averaged_probe_speeds = np.array([])

    return trial_averaged_beaconed_speeds, trial_averaged_non_beaconed_speeds, trial_averaged_probe_speeds

def bin_in_space(raw_position_data, processed_position_data, track_length):
    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/settings.vr_bin_size_cm)

    speeds_binned_in_space = []
    pos_binned_in_space = []
    acc_binned_in_space = []

    for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
        trial_x_position_cm = np.array(raw_position_data['x_position_cm'][np.array(raw_position_data['trial_number']) == trial_number], dtype="float64")
        trial_speeds = np.array(raw_position_data['speed_per200ms'][np.array(raw_position_data['trial_number']) == trial_number], dtype="float64")

        pos_bins = np.arange(0, track_length, settings.vr_bin_size_cm)# 1cm space bins

        if len(pos_bins)>1:
            # calculate the average speed and position in each space bin
            speed_bin_means, pos_bin_edges = np.histogram(trial_x_position_cm, pos_bins, weights=trial_speeds)
            speed_bin_means = speed_bin_means/np.histogram(trial_x_position_cm, pos_bins)[0]

            # get location bin centres
            pos_bin_centres = 0.5*(pos_bin_edges[1:]+pos_bin_edges[:-1])

            # and smooth
            speed_bin_means = convolve(speed_bin_means, gauss_kernel)

            # calculate the acceleration from the smoothed speed
            acceleration_space_bin_means = np.diff(np.array(speed_bin_means))
            acceleration_space_bin_means = np.hstack((0, acceleration_space_bin_means))

        else:
            speed_bin_means = []
            pos_bin_centres = []
            acceleration_space_bin_means = []

        speeds_binned_in_space.append(speed_bin_means)
        pos_binned_in_space.append(pos_bin_centres)
        acc_binned_in_space.append(acceleration_space_bin_means)

    processed_position_data["speeds_binned_in_space"] = speeds_binned_in_space
    processed_position_data["pos_binned_in_space"] = pos_binned_in_space
    processed_position_data["acc_binned_in_space"] = acc_binned_in_space

    return processed_position_data


def bin_in_time(raw_position_data, processed_position_data):
    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_time_seconds/settings.time_bin_size)

    speeds_binned_in_time = []
    pos_binned_in_time = []
    acc_binned_in_time = []

    for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
        trial_x_position_cm = np.array(raw_position_data['x_position_cm'][np.array(raw_position_data['trial_number']) == trial_number], dtype="float64")
        trial_speeds = np.array(raw_position_data['speed_per200ms'][np.array(raw_position_data['trial_number']) == trial_number], dtype="float64")
        trial_times = np.array(raw_position_data['time_seconds'][np.array(raw_position_data['trial_number']) == trial_number], dtype="float64")

        time_bins = np.arange(min(trial_times), max(trial_times), settings.time_bin_size)# 100ms time bins

        if len(time_bins)>1:
            # calculate the average speed and position in each 100ms time bin
            speed_time_bin_means = (np.histogram(trial_times, time_bins, weights = trial_speeds)[0] /
                                    np.histogram(trial_times, time_bins)[0])
            pos_time_bin_means = (np.histogram(trial_times, time_bins, weights = trial_x_position_cm)[0] /
                                  np.histogram(trial_times, time_bins)[0])

            # and smooth
            speed_time_bin_means = convolve(speed_time_bin_means, gauss_kernel)
            pos_time_bin_means = convolve(pos_time_bin_means, gauss_kernel)

            # calculate the acceleration from the smoothed speed
            acceleration_time_bin_means = np.diff(np.array(speed_time_bin_means))
            acceleration_time_bin_means = np.hstack((0, acceleration_time_bin_means))
        else:
            speed_time_bin_means = []
            pos_time_bin_means = []
            acceleration_time_bin_means = []

        speeds_binned_in_time.append(speed_time_bin_means)
        pos_binned_in_time.append(pos_time_bin_means)
        acc_binned_in_time.append(acceleration_time_bin_means)

    processed_position_data["speeds_binned_in_time"] = speeds_binned_in_time
    processed_position_data["pos_binned_in_time"] = pos_binned_in_time
    processed_position_data["acc_binned_in_time"] = acc_binned_in_time

    return processed_position_data



def process_position(raw_position_data, stop_threshold, track_length):
    processed_position_data = pd.DataFrame() # make dataframe for processed position data
    processed_position_data = bin_in_time(raw_position_data, processed_position_data)
    processed_position_data = bin_in_space(raw_position_data, processed_position_data, track_length)

    #TODO these functions should be removed and stops calculated from the time or space bins calculated above.
    #TODO speed plots will need to be changed accordingly.
    processed_position_data = PostSorting.vr_speed_analysis.process_speed(raw_position_data, processed_position_data, track_length)
    processed_position_data = PostSorting.vr_time_analysis.process_time(raw_position_data, processed_position_data,track_length)
    processed_position_data = PostSorting.vr_stop_analysis.process_stops(processed_position_data, stop_threshold, track_length)
    gc.collect()

    processed_position_data["new_trial_indices"] = raw_position_data["new_trial_indices"].dropna()

    print('-------------------------------------------------------------')
    print('position data processed')
    print('-------------------------------------------------------------')
    return processed_position_data


#  for testing
def main():
    print('-------------------------------------------------------------')

    processed_position_data = pd.read_pickle("/mnt/datastore/Harry/Cohort7_october2020/vr/M3_D7_2020-11-06_14-22-53"
                                             "/MountainSort/DataFrames/processed_position_data.pkl")
    trial_average_speed(processed_position_data)

if __name__ == '__main__':
    main()
