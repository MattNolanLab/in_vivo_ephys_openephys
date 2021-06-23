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
import os
import traceback
import warnings
import matplotlib.ticker as ticker
import sys
import plot_utility
import settings
import matplotlib.pylab as plt
import matplotlib as mpl
warnings.filterwarnings('ignore')


def calculate_grid_field_com(cluster_spike_data, position_data):
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
    firing_rate_maps = []

    firing_times=cluster_spike_data.firing_times/(settings.sampling_rate/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    if len(firing_times)==0:
        firing_rate_maps = np.zeros(int(settings.track_length))
        return firing_field_com, firing_field_com_trial_numbers, firing_field_com_trial_types, firing_rate_maps

    trial_numbers = np.array(position_data['trial_number'].to_numpy())
    trial_types = np.array(position_data['trial_type'].to_numpy())
    time_seconds = np.array(position_data['time_seconds'].to_numpy())
    x_position_cm = np.array(position_data['x_position_cm'].to_numpy())
    x_position_cm_elapsed = x_position_cm+((trial_numbers-1)*settings.track_length)

    instantaneous_firing_rate_per_ms = extract_instantaneous_firing_rate_for_spike2(cluster_spike_data) # returns firing rate per millisecond time bin
    instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[0:len(x_position_cm)]

    if not (len(instantaneous_firing_rate_per_ms) == len(trial_numbers)):
        # 0 pad until it is the same size (padding with 0 hz firing rate
        instantaneous_firing_rate_per_ms = np.append(instantaneous_firing_rate_per_ms, np.zeros(len(trial_numbers)-len(instantaneous_firing_rate_per_ms)))

    max_distance_elapsed = settings.track_length*max(trial_numbers)
    numerator, bin_edges = np.histogram(x_position_cm_elapsed, bins=int(max_distance_elapsed/settings.vr_grid_analysis_bin_size), range=(0, max_distance_elapsed), weights=instantaneous_firing_rate_per_ms)
    denominator, bin_edges = np.histogram(x_position_cm_elapsed, bins=int(max_distance_elapsed/settings.vr_grid_analysis_bin_size), range=(0, max_distance_elapsed))
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    firing_rate_map = numerator/denominator

    local_maxima_bin_idx = signal.argrelextrema(firing_rate_map, np.greater)[0]
    global_maxima_bin_idx = np.nanargmax(firing_rate_map)
    global_maxima = firing_rate_map[global_maxima_bin_idx]
    field_threshold = 0.2*global_maxima

    for local_maximum_idx in local_maxima_bin_idx:
        neighbouring_local_mins = find_neighbouring_minima(firing_rate_map, local_maximum_idx)
        closest_minimum_bin_idx = neighbouring_local_mins[np.argmin(np.abs(neighbouring_local_mins-local_maximum_idx))]
        field_size_in_bins = neighbouring_local_mins[1]-neighbouring_local_mins[0]

        if firing_rate_map[local_maximum_idx] - firing_rate_map[closest_minimum_bin_idx] > field_threshold:
            #firing_field.append(neighbouring_local_mins)

            field =  firing_rate_map[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
            field_bins = bin_centres[neighbouring_local_mins[0]:neighbouring_local_mins[1]+1]
            field_weights = field/np.sum(field)
            field_com = np.sum(field_weights*field_bins)

            # reverse calculate the field_com in cm from track start
            trial_number = (field_com//settings.track_length)+1
            trial_type = stats.mode(trial_types[trial_numbers==trial_number])[0][0]
            field_com = field_com%settings.track_length

            firing_field_com.append(field_com)
            firing_field_com_trial_numbers.append(trial_number)
            firing_field_com_trial_types.append(trial_type)

    for trial_number in np.unique(trial_numbers):
        trial_x_position_cm = x_position_cm[trial_numbers==trial_number]
        trial_instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[trial_numbers==trial_number]

        numerator, bin_edges = np.histogram(trial_x_position_cm, bins=int(settings.track_length/settings.vr_grid_analysis_bin_size), range=(0, settings.track_length), weights=trial_instantaneous_firing_rate_per_ms)
        denominator, bin_edges = np.histogram(trial_x_position_cm, bins=int(settings.track_length/settings.vr_grid_analysis_bin_size), range=(0, settings.track_length))

        firing_rate_map = numerator/denominator
        firing_rate_maps.append(firing_rate_map)

    return firing_field_com, firing_field_com_trial_numbers, firing_field_com_trial_types, firing_rate_maps


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

def extract_instantaneous_firing_rate_for_spike2(cluster_data):
    firing_times=cluster_data.firing_times/(settings.sampling_rate/1000) # convert from samples to ms
    if isinstance(firing_times, pd.Series):
        firing_times = firing_times.iloc[0]
    bins = np.arange(0,np.max(firing_times)+2000, 1)
    instantaneous_firing_rate = np.histogram(firing_times, bins=bins, range=(0, max(bins)))[0]

    gauss_kernel = Gaussian1DKernel(5) # sigma = 200ms
    instantaneous_firing_rate = convolve(instantaneous_firing_rate, gauss_kernel)

    return instantaneous_firing_rate

def process_vr_grid(spike_data, position_data):

    fields_com_cluster = []
    fields_com_trial_numbers_cluster = []
    fields_com_trial_types_cluster = []
    firing_rate_maps_cluster = []

    minimum_distance_to_field_in_next_trial =[]
    fields_com_next_trial_type = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        fields_com, field_com_trial_numbers, field_com_trial_types, firing_rate_maps = calculate_grid_field_com(cluster_df, position_data)

        if len(firing_rate_maps)==0:
            print("stop here")

        next_trial_type_cluster = []
        minimum_distance_to_field_in_next_trial_cluster=[]

        for i in range(len(fields_com)):
            field = fields_com[i]
            trial_number=field_com_trial_numbers[i]
            trial_type = int(field_com_trial_types[i])

            trial_type_tmp = position_data["trial_type"].to_numpy()
            trial_number_tmp = position_data["trial_number"].to_numpy()

            fields_in_next_trial = np.array(fields_com)[np.array(field_com_trial_numbers) == int(trial_number+1)]
            fields_in_next_trial = fields_in_next_trial[(fields_in_next_trial>50) & (fields_in_next_trial<150)]

            if len(fields_in_next_trial)>0:
                next_trial_type = int(np.unique(trial_type_tmp[trial_number_tmp == int(trial_number+1)])[0])
                minimum_field_difference = min(np.abs(fields_in_next_trial-field))

                minimum_distance_to_field_in_next_trial_cluster.append(minimum_field_difference)
                next_trial_type_cluster.append(next_trial_type)
            else:
                minimum_distance_to_field_in_next_trial_cluster.append(np.nan)
                next_trial_type_cluster.append(np.nan)

        fields_com_cluster.append(fields_com)
        fields_com_trial_numbers_cluster.append(field_com_trial_numbers)
        fields_com_trial_types_cluster.append(field_com_trial_types)
        firing_rate_maps_cluster.append(firing_rate_maps)

        minimum_distance_to_field_in_next_trial.append(minimum_distance_to_field_in_next_trial_cluster)
        fields_com_next_trial_type.append(next_trial_type_cluster)

    spike_data["fields_com"] = fields_com_cluster
    spike_data["fields_com_trial_number"] = fields_com_trial_numbers_cluster
    spike_data["fields_com_trial_type"] = fields_com_trial_types_cluster
    spike_data["firing_rate_maps"] = firing_rate_maps_cluster

    spike_data["minimum_distance_to_field_in_next_trial"] = minimum_distance_to_field_in_next_trial
    spike_data["fields_com_next_trial_type"] = fields_com_next_trial_type

    return spike_data

def calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type):

    cluster_firing_com = np.array(cluster_df["fields_com"].iloc[0])
    cluster_firing_com_trial_types = np.array(cluster_df["fields_com_trial_type"].iloc[0])

    if trial_type == "beaconed":
        n_trials = processed_position_data.beaconed_total_trial_number.iloc[0]
        firing_com = cluster_firing_com[cluster_firing_com_trial_types == 0]
    elif trial_type == "non-beaconed":
        n_trials = processed_position_data.nonbeaconed_total_trial_number.iloc[0]
        firing_com = cluster_firing_com[cluster_firing_com_trial_types == 1]
    elif trial_type == "probe":
        n_trials = processed_position_data.probe_total_trial_number.iloc[0]
        firing_com = cluster_firing_com[cluster_firing_com_trial_types == 2]
    else:
        print("no valid trial type was given")

    if n_trials==0:
        return np.nan
    else:
        return len(firing_com)/n_trials

def process_vr_field_stats(spike_data, processed_position_data):
    n_beaconed_fields_per_trial = []
    n_nonbeaconed_fields_per_trial = []
    n_probe_fields_per_trial = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        n_beaconed_fields_per_trial.append(calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type="beaconed"))
        n_nonbeaconed_fields_per_trial.append(calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type="non-beaconed"))
        n_probe_fields_per_trial.append(calculate_n_fields_per_trial(cluster_df, processed_position_data, trial_type="probe"))

    spike_data["n_beaconed_fields_per_trial"] = n_beaconed_fields_per_trial
    spike_data["n_nonbeaconed_fields_per_trial"] = n_nonbeaconed_fields_per_trial
    spike_data["n_probe_fields_per_trial"] = n_probe_fields_per_trial

    return spike_data

def process_vr_field_distances(spike_data):
    distance_between_fields = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        cluster_firing_com = np.array(cluster_df["fields_com"].iloc[0])
        cluster_firing_com_trial_types = np.array(cluster_df["fields_com_trial_type"].iloc[0])
        cluster_firing_com_trial_numbers = np.array(cluster_df["fields_com_trial_number"].iloc[0])

        distance_covered = (cluster_firing_com_trial_numbers*settings.track_length)-settings.track_length #total elapsed distance
        cluster_firing_com = cluster_firing_com+distance_covered

        cluster_firing_com_distance_between = np.diff(cluster_firing_com)
        distance_between_fields.append(cluster_firing_com_distance_between)

    spike_data["distance_between_fields"] = distance_between_fields

    return spike_data

def find_paired_recording(recording_path, of_recording_path_list):
    mouse=recording_path.split("/")[-1].split("_")[0]
    training_day=recording_path.split("/")[-1].split("_")[1]

    for paired_recording in of_recording_path_list:
        paired_mouse=paired_recording.split("/")[-1].split("_")[0]
        paired_training_day=paired_recording.split("/")[-1].split("_")[1]

        if (mouse == paired_mouse) and (training_day == paired_training_day):
            return paired_recording, True

    return None, False

def find_set(a,b):
    return set(a) & set(b)

def plot_spatial_autocorrelogram(spike_data, output_path):
    print('plotting spike spatial autocorrelogram...')
    save_path = output_path + '/Figures/spatial_autocorrelograms'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    spatial_auto_peak = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data["firing_times"].iloc[0]

        if len(firing_times_cluster)>1:
            x_position_cluster = np.array(cluster_spike_data["x_position_cm"].iloc[0])
            trial_numbers = np.array(cluster_spike_data["trial_number"].iloc[0])
            lap_distance_covered = (trial_numbers*settings.track_length)-settings.track_length #total elapsed distance
            x_position_cluster = x_position_cluster+lap_distance_covered
            x_position_cluster = x_position_cluster[~np.isnan(x_position_cluster)]
            x_position_cluster_bins = np.floor(x_position_cluster).astype(int)

            autocorr_window_size = 400
            lags = np.arange(0, autocorr_window_size, 1).astype(int) # were looking at 10 timesteps back and 10 forward

            autocorrelogram = np.array([])
            for lag in lags:
                correlated = len(find_set(x_position_cluster_bins+lag, x_position_cluster_bins))
                autocorrelogram = np.append(autocorrelogram, correlated)

            #b, a = signal.butter(5, 1, 'low', analog = True) #first parameter is signal order and the second one refers to frequenc limit. I set limit 30 so that I can see only below 30 frequency signal component
            #autocorrelogram = signal.filtfilt(b, a, autocorrelogram)

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            ax.bar(lags[1:], autocorrelogram[1:], edgecolor="black", align="edge")
            plt.ylabel('Counts', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,400)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plot_utility.style_vr_plot(ax, x_max=max(autocorrelogram[1:]))
            plt.locator_params(axis = 'x', nbins  = 8)
            tick_spacing = 50
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

            peaks = signal.argrelextrema(autocorrelogram, np.greater, order=20)[0]
            ax.scatter(lags[peaks], autocorrelogram[peaks], marker="x", color="r")
            ax.scatter(lags[peaks][0], autocorrelogram[peaks][0], marker="x", color="b")
            spatial_auto_peak.append(peaks[0])

            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_spatial_autocorrelogram_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

    spike_data["spatial_autocorr_peak_cm"] = spatial_auto_peak
    return spike_data

def simulate_rate_map(track_length, firing_field_distance, bin_size_cm=1):

    # round to the nearest integer value
    firing_field_distance = int(np.round(firing_field_distance))

    # make an empty rate map 2x the size of the track length
    rate_map = np.zeros(int(track_length/bin_size_cm)*2)

    # we reconstruct 1 firing field, and we assume firing field is less than 40cm wide and has a sd of 3
    # this should really matter that much as the peaks should be the most important thing to correlate

    field_xs = np.arange(0, 40)
    guassian = np.exp(-np.power(field_xs - 20, 2.) / (2 * np.power(3, 2.)))

    #rate_map_i = int(firing_field_distance/bin_size_cm)-len(guassian)
    for j in range(len(guassian)):
        #m =
        rate_map[j] = guassian[j]


        rate_map_i = int(firing_field_distance/bin_size_cm)-len(guassian)

        if not (rate_map_i>j):
            rate_map[rate_map_i + j] = guassian[j]



    #for i in range(int((track_length*2)//firing_field_distance)): # repeat as many times there are firing fields in two track lengths





def calculate_trial_firing_correlation(spike_data, position_data, output_path):
    print('plotting spike spatial autocorrelogram...')
    save_path = output_path + '/Figures/trial_correlations'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    spatial_auto_peak = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times = cluster_spike_data["firing_times"].iloc[0]/(settings.sampling_rate/1000) # convert from samples to ms

        if len(firing_times)>1:
            trial_numbers = np.array(position_data['trial_number'].to_numpy())
            trial_types = np.array(position_data['trial_type'].to_numpy())
            time_seconds = np.array(position_data['time_seconds'].to_numpy())
            x_position_cm = np.array(position_data['x_position_cm'].to_numpy())
            x_position_cm_elapsed = x_position_cm+((trial_numbers-1)*settings.track_length)

            instantaneous_firing_rate_per_ms = extract_instantaneous_firing_rate_for_spike2(cluster_spike_data) # returns firing rate per millisecond time bin
            instantaneous_firing_rate_per_ms = instantaneous_firing_rate_per_ms[0:len(x_position_cm)]

            if not (len(instantaneous_firing_rate_per_ms) == len(trial_numbers)):
                # 0 pad until it is the same size (padding with 0 hz firing rate
                instantaneous_firing_rate_per_ms = np.append(instantaneous_firing_rate_per_ms, np.zeros(len(trial_numbers)-len(instantaneous_firing_rate_per_ms)))


            # get firing field distance
            firing_field_distance_cluster = cluster_spike_data["spatial_autocorr_peak_cm"].iloc[0]

            simulated_rate_map = simulate_rate_map(settings.track_length, firing_field_distance_cluster, bin_size_cm=1)


            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            tick_spacing = 50
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.ylabel('Trial number', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,settings.track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')


            for trial_number in np.unique(trial_numbers):
                instantaneous_firing_rate_per_ms_trial = instantaneous_firing_rate_per_ms[trial_numbers == trial_number]
                x_position_cm_trial = x_position_cm[trial_numbers == trial_number]

                numerator, bin_edges = np.histogram(x_position_cm_trial, bins=int(settings.track_length/1), range=(0, settings.track_length), weights=instantaneous_firing_rate_per_ms_trial)
                denominator, _ = np.histogram(x_position_cm_trial, bins=int(settings.track_length/1), range=(0, settings.track_length))

                trial_rate_map = numerator/denominator
                bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

                peaks = signal.argrelextrema(trial_rate_map, np.greater, order=20)[0]

                #ax.scatter(bin_centres[peaks], np.ones(len(bin_centres[peaks]))*trial_number, marker="x", color="r")

                if len(peaks)>0:
                    ax.scatter(bin_centres[peaks][-1], trial_number, marker="x", color="r")

            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_trial_correlations_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def plot_inter_field_distance_histogram(spike_data, output_path):
    print('plotting field com histogram...')
    tick_spacing = 100
    save_path = output_path + '/Figures/field_distances'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

        if len(firing_times_cluster)>1:

            norm = mpl.colors.Normalize(vmin=0, vmax=settings.track_length)
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
            cmap.set_array([])

            fig, ax = plt.subplots(dpi=200)
            loop_factor=3
            hist_cmap = plt.cm.get_cmap('viridis')
            cluster_firing_com_distances = np.array(spike_data["distance_between_fields"].iloc[cluster_index])
            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            cluster_firing_com_distances = np.append(cluster_firing_com_distances, np.nan)

            for i in range(int(settings.track_length/settings.vr_grid_analysis_bin_size)*loop_factor):
                mask = (cluster_firing_com > i*settings.vr_grid_analysis_bin_size) & \
                       (cluster_firing_com < (i+1)*settings.vr_grid_analysis_bin_size)
                cluster_firing_com_distances_bin_i = cluster_firing_com_distances[mask]
                cluster_firing_com_distances_bin_i = cluster_firing_com_distances_bin_i[~np.isnan(cluster_firing_com_distances_bin_i)]

                field_hist, bin_edges = np.histogram(cluster_firing_com_distances_bin_i,
                                                     bins=int(settings.track_length/settings.vr_grid_analysis_bin_size)*loop_factor,
                                                     range=[0, settings.track_length*loop_factor])

                if i == 0:
                    bottom = np.zeros(len(field_hist))

                ax.bar(bin_edges[:-1], field_hist, width=np.diff(bin_edges), bottom=bottom, edgecolor="black",
                       align="edge", color=hist_cmap(i/int(settings.track_length/settings.vr_grid_analysis_bin_size)*loop_factor))
                bottom += field_hist

            cbar = fig.colorbar(cmap)
            cbar.set_label('Location (cm)', rotation=90, labelpad=20)
            plt.ylabel('Field Counts', fontsize=12, labelpad = 10)
            plt.xlabel('Field to Field Distance (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,400)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            x_max = max(bottom)
            #plot_utility.style_track_plot(ax, 200)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_distance_hist_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def plot_field_com_histogram(spike_data, output_path):
    tick_spacing = 50

    print('plotting field com histogram...')
    save_path = output_path + '/Figures/field_distributions'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

        if len(firing_times_cluster)>1:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            field_hist, bin_edges = np.histogram(cluster_firing_com, bins=int(settings.track_length/settings.vr_grid_analysis_bin_size), range=[0, settings.track_length])
            ax.bar(bin_edges[:-1], field_hist/np.sum(field_hist), width=np.diff(bin_edges), edgecolor="black", align="edge")
            plt.ylabel('Field Density', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            field_hist = np.nan_to_num(field_hist)

            x_max = max(field_hist/np.sum(field_hist))
            plot_utility.style_track_plot(ax, 200)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_hist_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def min_max_normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def plot_firing_rate_maps_per_trial(spike_data, output_path):
    print('plotting trial firing rate maps...')
    save_path = output_path + '/Figures/firing_rate_maps_trials'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:

            x_max = max(np.array(spike_data.beaconed_trial_number.iloc[cluster_index])) + 1
            if x_max>100:
                spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
            else:
                spikes_on_track = plt.figure(figsize=(4,(x_max/20)))

            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            cluster_firing_maps = np.array(spike_data["firing_rate_maps"].iloc[cluster_index])
            where_are_NaNs = np.isnan(cluster_firing_maps)
            cluster_firing_maps[where_are_NaNs] = 0

            if len(cluster_firing_maps) == 0:
                print("stop here")

            cluster_firing_maps = min_max_normalize(cluster_firing_maps)

            cmap = plt.cm.get_cmap("jet")
            cmap.set_bad(color='white')
            bin_size = settings.vr_grid_analysis_bin_size

            tmp = []
            for i in range(len(cluster_firing_maps[0])):
                for j in range(int(settings.vr_grid_analysis_bin_size)):
                    tmp.append(cluster_firing_maps[:, i].tolist())
            cluster_firing_maps = np.array(tmp).T

            c = ax.imshow(cluster_firing_maps, interpolation='none', cmap=cmap, vmin=0, vmax=np.max(cluster_firing_maps), origin='lower')

            plt.ylabel('Trial Number', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0, settings.track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            #plot_utility.style_track_plot(ax, 200)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_firing_rate_map_trials_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def plot_field_com_histogram_radial(spike_data, output_path):
    print('plotting field com histogram...')
    save_path = output_path + '/Figures/field_distributions'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

        if len(firing_times_cluster)>1:
            ax = plt.subplot(111, polar=True)
            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            field_hist, bin_edges = np.histogram(cluster_firing_com, bins=int(settings.track_length/settings.vr_grid_analysis_bin_size), range=[0, settings.track_length])

            width = (2*np.pi) / len(bin_edges[:-1])
            field_hist = field_hist/np.sum(field_hist)
            bottom = 0.4
            field_hist = min_max_normlise(field_hist, 0, 1)
            y_max = max(field_hist)

            bin_edges = np.linspace(0.0, 2 * np.pi, len(bin_edges[:-1]), endpoint=False)

            ax.bar(np.pi, y_max, width=np.pi*2*(20/settings.track_length), color="DarkGreen", edgecolor=None, alpha=0.25, bottom=bottom)
            ax.bar(0, y_max, width=np.pi*2*(60/settings.track_length), color="black", edgecolor=None, alpha=0.25, bottom=bottom)

            ax.bar(bin_edges, field_hist, width=width, edgecolor="black", align="edge", bottom=bottom)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.grid(alpha=0)
            ax.set_yticklabels([])
            ax.set_ylim([0,y_max])
            ax.set_xticklabels(['0cm', '', '50cm', '', '100cm', '', '150cm', ''], fontsize=15)
            ax.xaxis.set_tick_params(pad=20)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_hist_radial_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def plot_field_centre_of_mass_on_track(spike_data, output_path, plot_trials=["beaconed", "non_beaconed", "probe"]):

    print('plotting field rastas...')
    save_path = output_path + '/Figures/field_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]
        if len(firing_times_cluster)>1:

            x_max = max(np.array(spike_data.beaconed_trial_number.iloc[cluster_index])) + 1
            if x_max>100:
                spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
            else:
                spikes_on_track = plt.figure(figsize=(4,(x_max/20)))

            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            cluster_firing_com_trial_numbers = np.array(spike_data["fields_com_trial_number"].iloc[cluster_index])
            cluster_firing_com_trial_types = np.array(spike_data["fields_com_trial_type"].iloc[cluster_index])

            if "beaconed" in plot_trials:
                ax.plot(cluster_firing_com[cluster_firing_com_trial_types == 0], cluster_firing_com_trial_numbers[cluster_firing_com_trial_types == 0], "s", color='Black', markersize=4)
            if "non_beaconed" in plot_trials:
                ax.plot(cluster_firing_com[cluster_firing_com_trial_types == 1], cluster_firing_com_trial_numbers[cluster_firing_com_trial_types == 1], "s", color='Red', markersize=4)
            if "probe" in plot_trials:
                ax.plot(cluster_firing_com[cluster_firing_com_trial_types == 2], cluster_firing_com_trial_numbers[cluster_firing_com_trial_types == 2], "s", color='Blue', markersize=4)

            #ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=3)
            plt.ylabel('Field COM on trials', fontsize=12, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
            plt.xlim(0,200)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            plot_utility.style_track_plot(ax, 200)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            if len(plot_trials)<3:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_Cluster_' + str(cluster_id) + "_" + str("_".join(plot_trials)) + '.png', dpi=200)
            else:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()

def min_max_normlise(array, min_val, max_val):
    normalised_array = ((max_val-min_val)*((array-min(array))/(max(array)-min(array))))+min_val
    return normalised_array

def plot_field_com_ring_attractor_radial(spike_data, of_spike_data, output_path):
    print('plotting field com histogram...')
    save_path = output_path + '/Figures/radial_field_distances'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        firing_times_cluster = spike_data.firing_times.iloc[cluster_index]

        estimated_grid_spacing = of_spike_data.grid_spacing.iloc[cluster_index]

        if len(firing_times_cluster)>1:
            ax = plt.subplot(111, polar=True)

            cluster_firing_com = np.array(spike_data["fields_com"].iloc[cluster_index])
            cluster_firing_com_distances = np.array(spike_data["distance_between_fields"].iloc[cluster_index])
            loop_factor = (max(cluster_firing_com_distances)//settings.track_length)+1

            field_hist, bin_edges = np.histogram(cluster_firing_com_distances,
                                                 bins=int((settings.track_length/settings.vr_grid_analysis_bin_size)*loop_factor),
                                                 range=[0, int(settings.track_length*loop_factor)])

            width = (2*np.pi) / (len(bin_edges[:-1])/loop_factor)
            field_hist = field_hist/np.sum(field_hist)
            bottom = 0.4
            field_hist = min_max_normlise(field_hist, 0, 1)
            y_max = max(field_hist)

            bin_edges = np.linspace(0.0, loop_factor*2*np.pi, int(len(bin_edges[:-1])), endpoint=False)

            cmap = plt.cm.get_cmap('viridis')

            #ax.bar(np.pi, y_max, width=np.pi*2*(20/settings.track_length), color="DarkGreen", edgecolor=None, alpha=0.25, bottom=bottom)
            #ax.bar(0, y_max, width=np.pi*2*(60/settings.track_length), color="black", edgecolor=None, alpha=0.25, bottom=bottom)

            for i in range(int(loop_factor)):
                ax.bar(bin_edges[int(i*(len(bin_edges)/loop_factor)): int((i+1)*(len(bin_edges)/loop_factor))],
                       field_hist[int(i*(len(bin_edges)/loop_factor)): int((i+1)*(len(bin_edges)/loop_factor))],
                       width=width, edgecolor="black", align="edge", bottom=bottom, color=cmap(i/loop_factor), alpha=0.6)

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.grid(alpha=0)
            ax.set_yticklabels([])
            ax.set_ylim([0,y_max])
            estimated_grid_spacing = np.round(estimated_grid_spacing/2, decimals=1)
            ax.set_xticklabels([str(np.round(estimated_grid_spacing, decimals=1))+"cm", "", "", "", "", "", "", ""], fontsize=15)
            ax.xaxis.set_tick_params(pad=20)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_fields_distance_hist_radial_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()


def process_recordings(vr_recording_path_list, of_recording_path_list):

    for recording in vr_recording_path_list:
        paired_recording, found_paired_recording = find_paired_recording(recording, of_recording_path_list)
        try:
            output_path = recording+'/'+settings.sorterName
            position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/position_data.pkl")
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")
            processed_position_data = pd.read_pickle(recording+"/MountainSort/DataFrames/processed_position_data.pkl")

            spike_data = process_vr_grid(spike_data, position_data)
            #spike_data = process_vr_field_stats(spike_data, processed_position_data)
            spike_data = process_vr_field_distances(spike_data)
            spike_data = plot_spatial_autocorrelogram(spike_data, output_path)
            spike_data = calculate_trial_firing_correlation(spike_data, position_data, output_path)

            plot_inter_field_distance_histogram(spike_data=spike_data, output_path=output_path)
            plot_field_com_histogram(spike_data=spike_data, output_path=output_path)
            plot_firing_rate_maps_per_trial(spike_data=spike_data, output_path=output_path)
            plot_field_com_histogram_radial(spike_data=spike_data, output_path=output_path)
            plot_field_centre_of_mass_on_track(spike_data=spike_data, output_path=output_path, plot_trials=["beaconed", "non_beaconed", "probe"])
            plot_field_centre_of_mass_on_track(spike_data=spike_data, output_path=output_path, plot_trials=["beaconed"])
            plot_field_centre_of_mass_on_track(spike_data=spike_data, output_path=output_path, plot_trials=["non_beaconed"])
            plot_field_centre_of_mass_on_track(spike_data=spike_data, output_path=output_path, plot_trials=["probe"])

            if found_paired_recording:
                of_spatial_firing = pd.read_pickle(paired_recording+"/MountainSort/DataFrames/spatial_firing.pkl")
                plot_field_com_ring_attractor_radial(spike_data=spike_data, of_spike_data=of_spatial_firing, output_path=output_path)


        #spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            print("successfully processed and saved vr_grid analysis on "+recording)
        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)



def main():
    print('-------------------------------------------------------------')

    # give a path for a directory of recordings or path of a single recording
    vr_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/test_recordings_vr") if f.is_dir()]
    of_path_list = [f.path for f in os.scandir("/mnt/datastore/Harry/test_recordings") if f.is_dir()]

    process_recordings(vr_path_list, of_path_list)
    print("look now`")


if __name__ == '__main__':
    main()
