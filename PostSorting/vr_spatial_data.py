import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.vr_sync_spatial_data
import traceback
import control_sorting_analysis
import warnings
from scipy import stats
import plot_utility
import os
import sys
import settings
from astropy.convolution import convolve, Gaussian1DKernel

def get_stop_threshold_and_track_length(recording_path):
    parameter_file_path = control_sorting_analysis.get_tags_parameter_file(recording_path)
    stop_threshold, track_length, _ = PostSorting.post_process_sorted_data_vr.process_running_parameter_tag(parameter_file_path)
    return stop_threshold, track_length

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


def bin_in_space(raw_position_data, processed_position_data, track_length, smoothen=True):
    if smoothen:
        suffix="_smoothed"
    else:
        suffix=""

    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/settings.vr_bin_size_cm)
    n_trials = max(raw_position_data["trial_number"])

    # extract spatial variables from raw position
    speeds = np.array(raw_position_data['speed_per200ms'], dtype="float64")
    trial_numbers_raw = np.array(raw_position_data['trial_number'], dtype=np.int64)
    x_position_elapsed_cm = (track_length*(trial_numbers_raw-1))+np.array(raw_position_data['x_position_cm'], dtype="float64")

    # calculate the average speed and position in each 1cm spatial bin
    spatial_bins = np.arange(0, (n_trials*track_length)+1, settings.vr_bin_size_cm) # 1 cm bins
    speed_space_bin_means = (np.histogram(x_position_elapsed_cm, spatial_bins, weights = speeds)[0] / np.histogram(x_position_elapsed_cm, spatial_bins)[0])
    pos_space_bin_means = (np.histogram(x_position_elapsed_cm, spatial_bins, weights = x_position_elapsed_cm)[0] / np.histogram(x_position_elapsed_cm, spatial_bins)[0])
    tn_space_bin_means = (np.histogram(x_position_elapsed_cm, spatial_bins, weights = trial_numbers_raw)[0] / np.histogram(x_position_elapsed_cm, spatial_bins)[0]).astype(np.int64)
    tn_space_bin_means = (((0.5*(spatial_bins[1:]+spatial_bins[:-1]))//track_length)+1).astype(np.int64) # uncomment to get nan values for portions of first and last trial

    # and smooth
    if smoothen:
        speed_space_bin_means = convolve(speed_space_bin_means, gauss_kernel)
        pos_space_bin_means = convolve(pos_space_bin_means, gauss_kernel)

    # calculate the acceleration from the smoothed speed
    acceleration_space_bin_means = np.diff(np.array(speed_space_bin_means))
    acceleration_space_bin_means = np.hstack((0, acceleration_space_bin_means))

    # recalculate the position from the elapsed distance
    pos_space_bin_means = pos_space_bin_means%track_length

    # create empty lists to be filled and put into processed_position_data
    speeds_binned_in_space = []; pos_binned_in_space = []; acc_binned_in_space = []

    for trial_number in range(1, n_trials+1):
        speeds_binned_in_space.append(speed_space_bin_means[tn_space_bin_means == trial_number].tolist())
        pos_binned_in_space.append(pos_space_bin_means[tn_space_bin_means == trial_number].tolist())
        acc_binned_in_space.append(acceleration_space_bin_means[tn_space_bin_means == trial_number].tolist())

    processed_position_data["speeds_binned_in_space"+suffix] = speeds_binned_in_space
    processed_position_data["pos_binned_in_space"+suffix] = pos_binned_in_space
    processed_position_data["acc_binned_in_space"+suffix] = acc_binned_in_space

    return processed_position_data


def bin_in_time(raw_position_data, processed_position_data, track_length, smoothen=True):
    if smoothen:
        suffix="_smoothed"
    else:
        suffix=""

    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_time_seconds/settings.time_bin_size)
    n_trials = max(raw_position_data["trial_number"])

    # extract spatial variables from raw position
    speeds = np.array(raw_position_data['speed_per200ms'], dtype="float64")
    times = np.array(raw_position_data['time_seconds'], dtype="float64")
    trial_numbers_raw = np.array(raw_position_data['trial_number'], dtype=np.int64)
    x_position_elapsed_cm = (track_length*(trial_numbers_raw-1))+np.array(raw_position_data['x_position_cm'], dtype="float64")

    # calculate the average speed and position in each 100ms time bin
    time_bins = np.arange(min(times), max(times), settings.time_bin_size) # 100ms time bins
    speed_time_bin_means = (np.histogram(times, time_bins, weights = speeds)[0] / np.histogram(times, time_bins)[0])
    pos_time_bin_means = (np.histogram(times, time_bins, weights = x_position_elapsed_cm)[0] / np.histogram(times, time_bins)[0])
    tn_time_bin_means = (np.histogram(times, time_bins, weights = trial_numbers_raw)[0] / np.histogram(times, time_bins)[0]).astype(np.int64)

    # and smooth
    if smoothen:
        speed_time_bin_means = convolve(speed_time_bin_means, gauss_kernel)
        pos_time_bin_means = convolve(pos_time_bin_means, gauss_kernel)

    # calculate the acceleration from the smoothed speed
    acceleration_time_bin_means = np.diff(np.array(speed_time_bin_means))
    acceleration_time_bin_means = np.hstack((0, acceleration_time_bin_means))

    # recalculate the position from the elapsed distance
    pos_time_bin_means = pos_time_bin_means%track_length

    # create empty lists to be filled and put into processed_position_data
    speeds_binned_in_time = []; pos_binned_in_time = []; acc_binned_in_time = []

    for trial_number in range(1, n_trials+1):
        speeds_binned_in_time.append(speed_time_bin_means[tn_time_bin_means == trial_number].tolist())
        pos_binned_in_time.append(pos_time_bin_means[tn_time_bin_means == trial_number].tolist())
        acc_binned_in_time.append(acceleration_time_bin_means[tn_time_bin_means == trial_number].tolist())

    processed_position_data["speeds_binned_in_time"+suffix] = speeds_binned_in_time
    processed_position_data["pos_binned_in_time"+suffix] = pos_binned_in_time
    processed_position_data["acc_binned_in_time"+suffix] = acc_binned_in_time
    return processed_position_data


def add_trial_variables(raw_position_data, processed_position_data, track_length):
    n_trials = max(raw_position_data["trial_number"])

    trial_numbers = []
    trial_types = []
    position_bin_centres = []
    for trial_number in range(1, n_trials+1):
        trial_type = int(stats.mode(np.array(raw_position_data['trial_type'][np.array(raw_position_data['trial_number']) == trial_number]), axis=None)[0])
        bins = np.arange(0, track_length+1, settings.vr_bin_size_cm)
        bin_centres = 0.5*(bins[1:]+bins[:-1])

        trial_numbers.append(trial_number)
        trial_types.append(trial_type)
        position_bin_centres.append(bin_centres)

    processed_position_data["trial_number"] = trial_numbers
    processed_position_data["trial_type"] = trial_types
    processed_position_data["position_bin_centres"] = position_bin_centres
    return processed_position_data


def process_position(raw_position_data, stop_threshold, track_length):
    processed_position_data = pd.DataFrame() # make dataframe for processed position data
    processed_position_data = add_trial_variables(raw_position_data, processed_position_data, track_length)
    processed_position_data = bin_in_time(raw_position_data, processed_position_data, track_length, smoothen=True)
    processed_position_data = bin_in_time(raw_position_data, processed_position_data, track_length, smoothen=False)
    processed_position_data = bin_in_space(raw_position_data, processed_position_data, track_length, smoothen=True)
    processed_position_data = bin_in_space(raw_position_data, processed_position_data, track_length, smoothen=False)
    processed_position_data = PostSorting.vr_stop_analysis.process_stops(processed_position_data, stop_threshold, track_length)
    gc.collect()

    processed_position_data["new_trial_indices"] = raw_position_data["new_trial_indices"].dropna()

    print('-------------------------------------------------------------')
    print('position data processed')
    print('-------------------------------------------------------------')
    return processed_position_data


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]

def smooth_raw_x_position(raw_position_data, track_length):
    # this function remakes the raw position data by smoothening the blender output
    # this is important when a mice run very fast and and can skip bins due to the 60Hz blender sampling rate

    rpd = np.asarray(raw_position_data["x_position_cm"])
    trial_numbers_raw = np.array(raw_position_data['trial_number'], dtype=np.int64)
    rpd_elapsed = (track_length*(trial_numbers_raw-1))+rpd
    gauss_kernel = Gaussian1DKernel(stddev=200)
    rpd_elapsed = convolve(rpd_elapsed, gauss_kernel)
    rpd_elapsed = moving_sum(rpd_elapsed, window=100)/100
    rpd_elapsed = np.append(rpd_elapsed, np.zeros(len(raw_position_data["x_position_cm"])-len(rpd_elapsed)))
    rpd = rpd_elapsed%track_length
    raw_position_data["x_position_cm"] = rpd
    return raw_position_data

def process_recordings(vr_recording_path_list):
    vr_recording_path_list.sort()

    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            output_path = recording+'/'+settings.sorterName
            stop_threshold, track_length = get_stop_threshold_and_track_length(recording)
            raw_position_data, position_data = PostSorting.vr_sync_spatial_data.syncronise_position_data(recording, output_path, track_length)
            raw_position_data = smooth_raw_x_position(raw_position_data, track_length)
            processed_position_data = process_position(raw_position_data, stop_threshold, track_length)
            processed_position_data.to_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")
            position_data.to_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            print("successfully processed on "+recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)


#  for testing
def main():
    print('-------------------------------------------------------------')
    vr_path_list = []
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort6_july2020/vr") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort8_may2021/vr") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort9_Junji/vr") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/vr") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/VirtualReality") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/VirtualReality") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort2/VirtualReality") if f.is_dir()])
    #vr_path_list = ["/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality/M1_D20_2018-10-15_11-51-45",
    #                "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/VirtualReality/M2_D20_2019-03-29_13-03-54",
    #                "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality/M1_D28_2018-10-27_12-14-26",
    #                "/mnt/datastore/Harry/Cohort7_october2020/vr/M7_D4_2020-11-01_16-05-43"]
    process_recordings(vr_path_list)

    print("processed_position_data dataframes have been remade")

if __name__ == '__main__':
    main()
