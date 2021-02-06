import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_time_analysis
import PostSorting.vr_make_plots
import PostSorting.vr_cued
import PostSorting.theta_modulation
from scipy import stats
from scipy import signal
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel


def calculate_grid_field_com(cluster_spike_data, position_data, bin_size, prm):
    '''
    :param spike_data:
    :param prm:
    :return:

    for each trial of each trial type we want to
    calculate the centre of mass of all detected field
    centre of mass is defined as

    '''

    firing_field_com = []
    firing_field_com_trial_numbers = []
    firing_field_com_trial_types = []
    #firing_field = []


    trial_numbers = np.array(position_data['trial_number'].to_numpy())
    trial_types = np.array(position_data['trial_type'].to_numpy())
    time_seconds = np.array(position_data['time_seconds'].to_numpy())
    x_position_cm = np.array(position_data['x_position_cm'].to_numpy())

    instantaneous_firing_rate_per_ms = extract_instantaneous_firing_rate_for_spike2(cluster_spike_data, prm) # returns firing rate per millisecond time bin
    instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[0:len(x_position_cm)]

    if not (len(instantaneous_firing_rate_per_ms) == len(trial_numbers)):
        # 0 pad until it is the same size (padding with 0 hz firing rate
        instantaneous_firing_rate_per_ms = np.append(instantaneous_firing_rate_per_ms, np.zeros(len(trial_numbers)-len(instantaneous_firing_rate_per_ms)))

    for trial_number in np.unique(trial_numbers):
        trial_type = stats.mode(trial_types[trial_numbers==trial_number])[0][0]
        trial_x_position_cm = x_position_cm[trial_numbers==trial_number]
        trial_instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[trial_numbers==trial_number]

        numerator, bin_edges = np.histogram(trial_x_position_cm, bins=int(prm.get_track_length()/bin_size), range=(0, prm.track_length), weights=trial_instantaneous_firing_rate_per_ms)
        denominator, bin_edges = np.histogram(trial_x_position_cm, bins=int(prm.get_track_length()/bin_size), range=(0, prm.track_length))
        bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

        firing_rate_map = numerator/denominator

        local_maxima_bin_idx = signal.argrelextrema(firing_rate_map, np.greater)[0]
        global_maxima_bin_idx = np.nanargmax(firing_rate_map)
        global_maxima = firing_rate_map[global_maxima_bin_idx]

        field_threshold = 0.2*global_maxima

        for local_maximum_idx in local_maxima_bin_idx:
            neighbouring_local_mins = find_neighbouring_minima(firing_rate_map, local_maximum_idx)
            closest_minimum_bin_idx = neighbouring_local_mins[np.argmin(np.abs(neighbouring_local_mins-local_maximum_idx))]

            if firing_rate_map[local_maximum_idx] - firing_rate_map[closest_minimum_bin_idx] > field_threshold:
                #firing_field.append(neighbouring_local_mins)

                field =  firing_rate_map[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
                field_bins = bin_centres[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
                field_weights = field/np.sum(field)

                field_com = np.sum(field_weights*field_bins)
                firing_field_com.append(field_com)
                firing_field_com_trial_numbers.append(trial_number)
                firing_field_com_trial_types.append(trial_type)

    return firing_field_com, firing_field_com_trial_numbers, firing_field_com_trial_types


def find_neighbouring_minima(firing_rate_map, local_maximum_idx):
    # walk right
    local_min_right = local_maximum_idx
    local_min_right_found = False
    for i in np.arange(local_maximum_idx, len(firing_rate_map)): #local max to end
        if local_min_right_found == False:
            if np.isnan(firing_rate_map[i]):
                continue
            elif firing_rate_map[i] < firing_rate_map[local_min_right]:
                local_min_right = i
            elif firing_rate_map[i] > firing_rate_map[local_min_right]:
                local_min_right_found = True

    # walk left
    local_min_left = local_maximum_idx
    local_min_left_found = False
    for i in np.arange(0, local_maximum_idx)[::-1]: # local max to start
        if local_min_left_found == False:
            if np.isnan(firing_rate_map[i]):
                continue
            elif firing_rate_map[i] < firing_rate_map[local_min_left]:
                local_min_left = i
            elif firing_rate_map[i] > firing_rate_map[local_min_left]:
                local_min_left_found = True

    return (local_min_left, local_min_right)


def extract_instantaneous_firing_rate_for_spike(cluster_data, prm):
    firing_times=cluster_data.firing_times/(prm.get_sampling_rate()/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    bins = np.arange(0,np.max(firing_times)+500, 1)
    instantaneous_firing_rate = np.histogram(firing_times, bins=bins, range=(0, max(bins)))[0]

    gauss_kernel = Gaussian1DKernel(5) # sigma = 200ms
    smoothened_instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)

    inds = np.digitize(firing_times, bins)

    ifr = []
    for i in inds:
        ifr.append(smoothened_instantaneous_firing_rate[i-1])

    smoothened_instantaneous_firing_rate_per_spike = np.array(ifr)
    return smoothened_instantaneous_firing_rate_per_spike

def extract_instantaneous_firing_rate_for_spike2(cluster_data, prm):
    firing_times=cluster_data.firing_times/(prm.get_sampling_rate()/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    bins = np.arange(0,np.max(firing_times)+2000, 1)
    instantaneous_firing_rate = np.histogram(firing_times, bins=bins, range=(0, max(bins)))[0]

    gauss_kernel = Gaussian1DKernel(5) # sigma = 200ms
    smoothened_instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)

    return smoothened_instantaneous_firing_rate

def process_vr_grid(spike_data, position_data, bin_size, prm):

    fields_com_cluster = []
    fields_com_trial_numbers_cluster = []
    fields_com_trial_types_cluster = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        fields_com, field_com_trial_numbers, field_com_trial_types = calculate_grid_field_com(cluster_df, position_data, bin_size, prm)
        fields_com_cluster.append(fields_com)
        fields_com_trial_numbers_cluster.append(field_com_trial_numbers)
        fields_com_trial_types_cluster.append(field_com_trial_types)

    spike_data["fields_com"] = fields_com_cluster
    spike_data["fields_com_trial_number"] = fields_com_trial_numbers_cluster
    spike_data["fields_com_trial_type"] = fields_com_trial_types_cluster
    return spike_data


#  for testing
def main():
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()
    params.set_sampling_rate(30000)
    bin_size = 20 # cm

    params.set_output_path("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted/M2_D11_2019-07-01_13-50-47/MountainSort")
    position_data = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted/M2_D11_2019-07-01_13-50-47/MountainSort/DataFrames/position_data.pkl")
    spike_data = pd.read_pickle("/mnt/datastore/Harry/Mouse_data_for_sarah_paper/_cohort5/VirtualReality/M2_sorted/M2_D11_2019-07-01_13-50-47/MountainSort/DataFrames/spatial_firing.pkl")
    spike_data = process_vr_grid(spike_data, position_data, bin_size, params)

    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed", "non_beaconed", "probe"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["non_beaconed"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["probe"])


    params.set_output_path("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D6_2020-08-10_14-17-21/MountainSort")
    position_data = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D6_2020-08-10_14-17-21/MountainSort/DataFrames/position_data.pkl")
    spike_data = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D6_2020-08-10_14-17-21/MountainSort/DataFrames/spatial_firing.pkl")
    spike_data = process_vr_grid(spike_data, position_data, bin_size, params)

    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed", "non_beaconed", "probe"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["non_beaconed"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["probe"])



    params.set_output_path("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D7_2020-08-11_14-49-23/MountainSort")
    position_data = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D7_2020-08-11_14-49-23/MountainSort/DataFrames/position_data.pkl")
    spike_data = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D7_2020-08-11_14-49-23/MountainSort/DataFrames/spatial_firing.pkl")
    spike_data = process_vr_grid(spike_data, position_data, bin_size, params)

    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed", "non_beaconed", "probe"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["non_beaconed"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["probe"])



    params.set_output_path("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D8_2020-08-12_15-06-01/MountainSort")
    position_data = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D8_2020-08-12_15-06-01/MountainSort/DataFrames/position_data.pkl")
    spike_data = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D8_2020-08-12_15-06-01/MountainSort/DataFrames/spatial_firing.pkl")
    spike_data = process_vr_grid(spike_data, position_data, bin_size, params)

    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed", "non_beaconed", "probe"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["non_beaconed"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["probe"])



    params.set_output_path("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D9_2020-08-13_15-16-48/MountainSort")
    position_data = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D9_2020-08-13_15-16-48/MountainSort/DataFrames/position_data.pkl")
    spike_data = pd.read_pickle("/mnt/datastore/Harry/Cohort6_july2020/vr/M1_D9_2020-08-13_15-16-48/MountainSort/DataFrames/spatial_firing.pkl")
    spike_data = process_vr_grid(spike_data, position_data, bin_size, params)

    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed", "non_beaconed", "probe"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["beaconed"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["non_beaconed"])
    PostSorting.vr_make_plots.plot_field_centre_of_mass_on_track(spike_data=spike_data, prm=params, plot_trials=["probe"])

    print("look now`")


if __name__ == '__main__':
    main()
