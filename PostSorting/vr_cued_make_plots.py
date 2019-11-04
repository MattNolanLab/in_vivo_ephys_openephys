import os
import matplotlib.pylab as plt
import plot_utility
import numpy as np
import PostSorting.vr_stop_analysis
import PostSorting.vr_extract_data
import PostSorting.vr_cued
from numpy import inf
import gc
import math
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.colorbar as cbar
import pylab as pl
from scipy import stats

'''

This file deals with plotting of a cue conditioned rewarded location variant of the classic distance estimation task 
described in Tennant et al. 2018. 

# Plot basic info to check recording is good:
> movement channel
> trial channels (one and two)

'''

def load_stop_data(spatial_data):
    locations = spatial_data['stop_location_cm'].values
    trials = spatial_data['stop_trial_number'].values
    trial_type = spatial_data['stop_trial_type'].values
    return locations,trials,trial_type

def load_first_stop_data(spatial_data):
    locations = spatial_data['first_series_location_cm'].values
    trials = spatial_data['first_series_trial_number'].values
    trial_type = spatial_data['first_series_trial_type'].values
    return locations,trials,trial_type

def split_stop_data_by_trial_type(spatial_data, first_stops=False):
    if first_stops:
        locations, trials, trial_type = load_first_stop_data(spatial_data)
    else:
        locations,trials,trial_type = load_stop_data(spatial_data)

    stop_data=np.transpose(np.vstack((locations, trials, trial_type)))
    beaconed = np.delete(stop_data, np.where(stop_data[:,2]>0),0)
    nonbeaconed = np.delete(stop_data, np.where(stop_data[:,2]!=1),0)
    probe = np.delete(stop_data, np.where(stop_data[:,2]!=2),0)
    return beaconed, nonbeaconed, probe


def plot_stops_on_track_offset(raw_position_data, processed_position_data, prm):
    print('I am plotting stop rasta offset from the goal location...')
    save_path = prm.get_output_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed,nonbeaconed,probe = split_stop_data_by_trial_type(processed_position_data)
    #reward_locs = np.array(processed_position_data.rewarded_stop_locations)
    #reward_trials = np.array(processed_position_data.rewarded_trials)

    trial_bb_start, trial_bb_end = find_blackboxes_to_plot(raw_position_data, prm)
    fill_blackbox(trial_bb_start, ax)
    fill_blackbox(trial_bb_end, ax)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='0.5', markersize=2)
    ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='red', markersize=2)
    #ax.plot(probe[:,0], probe[:,1], 'o', color='blue', markersize=2)
    #ax.plot(reward_locs, reward_trials, '>', color='Red', markersize=3)
    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    #plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(-200,200)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot_cue_conditioned(ax, prm.get_track_length())
    x_max = max(raw_position_data.trial_number) + 0.5
    plot_utility.style_vr_plot_offset(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_output_path() + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()

def find_blackboxes_to_plot(raw_position_data, prm):
    trial_bb_start = []
    trial_bb_end = []
    for trial_number in range(1, max(raw_position_data["trial_number"]) + 1):
        trial_goal_pos = np.asarray(raw_position_data["goal_location_cm"][raw_position_data["trial_number"] == trial_number])[0]
        trial_bb_start.append(15-trial_goal_pos)
        trial_bb_end.append(285-trial_goal_pos)

    # returns the centres of the black boxes for each trial
    return trial_bb_start, trial_bb_end

def fill_blackbox(blackbox_centres, ax):
    # remove last 2 trials in case of inaccuracies as is inaccurate
    og_blackbox_centres_len = len(blackbox_centres)
    blackbox_centres = blackbox_centres[0:-2]

    # check if blackboxes are all in same place
    if np.std(blackbox_centres) > 5:
        # fills in black boxes per trial
        for trial_number in range(1, len(blackbox_centres)+1):
            x = [blackbox_centres[trial_number - 1]-15,
                 blackbox_centres[trial_number - 1]+15,
                 blackbox_centres[trial_number - 1]+15,
                 blackbox_centres[trial_number - 1]-15]
            y = [trial_number-0.5, trial_number-0.5, trial_number+0.5, trial_number+0.5]
            ax.fill(x, y, alpha=0.25, color="k")
    else:
        mean_pos = np.mean(blackbox_centres)
        x = [mean_pos - 15, mean_pos + 15, mean_pos + 15, mean_pos - 15]
        y = [0.5, 0.5, og_blackbox_centres_len, og_blackbox_centres_len]
        ax.fill(x, y, alpha=0.25, color="k")

    return ax

def plot_stop_histogram(raw_position_data, processed_position_data, prm):
    print('plotting stop histogram...')
    save_path = prm.get_output_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    position_bins = np.array(processed_position_data["position_bins"])
    average_stops = np.array(processed_position_data["average_stops"])
    ax.plot(position_bins,average_stops, '-', color='Black')
    plt.ylabel('Stops (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,prm.get_track_length())
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, prm.get_track_length())
    x_max = max(processed_position_data.average_stops)+0.1
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_output_path() + '/Figures/behaviour/stop_histogram' + '.png', dpi=200)
    plt.close()


def plot_speed_histogram(raw_position_data, processed_position_data, prm):
    print('plotting speed histogram...')
    save_path = prm.get_output_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    speed_histogram = plt.figure(figsize=(6,4))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    position_bins = np.array(processed_position_data["position_bins"].dropna(axis=0))
    average_speed = np.array(processed_position_data["binned_speed_ms"].dropna(axis=0))
    ax.plot(position_bins,average_speed, '-', color='Black')
    plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,prm.get_track_length())
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, prm.get_track_length())
    x_max = max(processed_position_data.binned_speed_ms)+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_output_path() + '/Figures/behaviour/speed_histogram' + '.png', dpi=200)
    plt.close()


'''
# Plot spatial firing info:
> spikes per trial
> firing rate

'''
def plot_spikes_on_track_cue_offset(spike_data,raw_position_data,processed_position_data, prm, prefix):
    # only called for cue conditioning PI task
    print('plotting spike rastas with cue offsets...')
    save_path = prm.get_output_path() + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    #rewarded_locations = np.array(processed_position_data['rewarded_stop_locations'].dropna(axis=0))  #
    #rewarded_trials = np.array(processed_position_data['rewarded_trials'].dropna(axis=0))

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        x_max = max(np.array(spike_data.at[cluster_index, 'beaconed_trial_number'])) + 1
        spikes_on_track = plt.figure(figsize=(4, (x_max / 32)))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        # fill in black box locations
        trial_bb_start, trial_bb_end = find_blackboxes_to_plot(raw_position_data, prm)
        fill_blackbox(trial_bb_start, ax)
        fill_blackbox(trial_bb_end, ax)

        ax.plot(spike_data.loc[cluster_index].beaconed_position_cm,
                spike_data.loc[cluster_index].beaconed_trial_number,
                '|', color='Black', markersize=4)
        ax.plot(spike_data.loc[cluster_index].nonbeaconed_position_cm,
                spike_data.loc[cluster_index].nonbeaconed_trial_number, '|', color='Red', markersize=4)
        #ax.plot(spike_data.loc[cluster_index].probe_position_cm, spike_data.loc[cluster_index].probe_trial_number,
        #        '|',
        #        color='Blue', markersize=4)
        #ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=3)

        plt.ylabel('Spikes on trials', fontsize=12, labelpad=10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad=10)
        #plt.xlim(0, 200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plot_utility.style_track_plot_cue_conditioned(ax, prm.get_track_length())
        plot_utility.style_vr_plot_offset(ax, x_max)
        plt.locator_params(axis='y', nbins=4)
        try:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        except ValueError:
            continue
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_track_firing_Cluster_' + str(cluster_index + 1) + '.png', dpi=200)
        plt.close()


def plot_spikes_on_track_cue_offset_order(spike_data,raw_position_data,processed_position_data, prm, prefix):
    # only called for cue conditioning PI task
    print('plotting spike rastas with cue offsets...')
    save_path = prm.get_output_path() + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    #rewarded_locations = np.array(processed_position_data['rewarded_stop_locations'].dropna(axis=0))  #
    #rewarded_trials = np.array(processed_position_data['rewarded_trials'].dropna(axis=0))

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        x_max = max(np.array(spike_data.at[cluster_index, 'beaconed_trial_number'])) + 1
        spikes_on_track = plt.figure(figsize=(4, (x_max / 32)))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        # fill in black box locations
        trial_bb_start, trial_bb_end = find_blackboxes_to_plot(raw_position_data, prm)

        beaconed = np.array([spike_data.loc[cluster_index].beaconed_position_cm,
                             spike_data.loc[cluster_index].beaconed_trial_number,
                             np.zeros(len(spike_data.loc[cluster_index].beaconed_trial_number))]).transpose()

        nonbeaconed = np.array([spike_data.loc[cluster_index].nonbeaconed_position_cm,
                                 spike_data.loc[cluster_index].nonbeaconed_trial_number,
                                 np.zeros(len(spike_data.loc[cluster_index].nonbeaconed_trial_number))]).transpose()

        probe = np.array([0])

        beaconed, nonbeaconed, probe, trial_bb_start, trial_bb_end = order_by_cue(beaconed, nonbeaconed, probe,
                                                                                  trial_bb_start, trial_bb_end)

        fill_blackbox(trial_bb_start, ax)
        fill_blackbox(trial_bb_end, ax)

        # uncomment if you want to plot stops
        # ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)
        # ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)
        # ax.plot(probe[:,0], probe[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)

        ax.plot(beaconed[:,0], beaconed[:,1], '|', color='Black', markersize=4)
        ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], '|', color='Red', markersize=4)
        #ax.plot(spike_data.loc[cluster_index].probe_position_cm, spike_data.loc[cluster_index].probe_trial_number,
        #        '|',
        #        color='Blue', markersize=4)
        #ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=3)

        plt.ylabel('Spikes on trials', fontsize=12, labelpad=10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad=10)
        #plt.xlim(0, 200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plot_utility.style_track_plot_cue_conditioned(ax, prm.get_track_length())
        plot_utility.style_vr_plot_offset(ax, x_max)
        plt.locator_params(axis='y', nbins=4)
        try:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        except ValueError:
            continue
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_track_firing_orded_Cluster_' + str(cluster_index + 1) + '.png', dpi=200)
        plt.close()


def plot_firing_rate_maps(spike_data, prm, prefix):
    print('I am plotting firing rate maps...')
    save_path = prm.get_output_path() + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        avg_spikes_on_track = plt.figure(figsize=(6,4))

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate = PostSorting.vr_extract_data.extract_smoothed_average_firing_rate_data(spike_data, cluster_index)

        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(avg_beaconed_spike_rate, '-', color='Black')
        ax.plot(avg_nonbeaconed_spike_rate, '-', color='Red')
        ax.plot(avg_probe_spike_rate, '-', color='Blue')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        avg_beaconed_spike_rate[avg_beaconed_spike_rate == inf] = 0
        avg_nonbeaconed_spike_rate[avg_nonbeaconed_spike_rate == inf] = 0
        avg_probe_spike_rate[avg_probe_spike_rate == inf] = 0
        nb_x_max = np.nanmax(avg_beaconed_spike_rate)
        b_x_max = np.nanmax(avg_nonbeaconed_spike_rate)
        p_x_max = np.nanmax(avg_probe_spike_rate)
        if b_x_max > nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, b_x_max)
        elif b_x_max < nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, nb_x_max)
        elif b_x_max > nb_x_max and b_x_max < p_x_max:
            plot_utility.style_vr_plot(ax, p_x_max)
        plot_utility.style_track_plot(ax, 200)
        plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.15, left=0.12, right=0.87, top=0.92)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_rate_map_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()


def plot_gc_firing_rate_maps(spike_data, prm, prefix):
    print('I am plotting firing rate maps...')
    save_path = prm.get_output_path() + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        avg_spikes_on_track = plt.figure(figsize=(6,4))

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate = PostSorting.vr_extract_data.extract_gc_firing_rate_data(spike_data, cluster_index)

        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(avg_beaconed_spike_rate, '-', color='Black')
        ax.plot(avg_nonbeaconed_spike_rate, '-', color='Red')
        ax.plot(avg_probe_spike_rate, '-', color='Blue')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        nb_x_max = np.nanmax(avg_beaconed_spike_rate)
        b_x_max = np.nanmax(avg_nonbeaconed_spike_rate)
        p_x_max = np.nanmax(avg_probe_spike_rate)
        if b_x_max > nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, b_x_max)
        elif b_x_max < nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, nb_x_max)
        elif b_x_max > nb_x_max and b_x_max < p_x_max:
            plot_utility.style_vr_plot(ax, p_x_max)
        plot_utility.style_track_plot(ax, 200)
        plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.15, left=0.12, right=0.87, top=0.92)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_rate_map_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()


'''
plot gaussian convolved firing rate in time against similarly convolved speed and location. 
'''

def plot_convolved_rates_in_time(spike_data, prm):
    print('plotting spike rastas...')
    save_path = prm.get_output_path() + '/Figures/ConvolvedRates_InTime'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        spikes_on_track = plt.figure(figsize=(4,5))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        firing_rate = spike_data.loc[cluster_index].spike_rate_in_time
        speed = spike_data.loc[cluster_index].speed_rate_in_time
        x_max= np.max(firing_rate)
        ax.plot(firing_rate, speed, '|', color='Black', markersize=4)
        plt.ylabel('Firing rate (Hz)', fontsize=12, labelpad = 10)
        plt.xlabel('Speed (cm/sec)', fontsize=12, labelpad = 10)
        #plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #plot_utility.style_track_plot(ax, 200)
        #plot_utility.style_vr_plot(ax, x_max)
        plt.locator_params(axis = 'y', nbins  = 4)
        try:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        except ValueError:
            continue
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_rate_versus_SPEED_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()

        spikes_on_track = plt.figure(figsize=(4,5))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        position = spike_data.loc[cluster_index].position_rate_in_time
        ax.plot(firing_rate, position, '|', color='Black', markersize=4)
        plt.ylabel('Firing rate (Hz)', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        # ]polt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #plot_utility.style_track_plot(ax, 200)
        #plot_utility.style_vr_plot(ax, x_max)
        plt.locator_params(axis = 'y', nbins  = 4)
        try:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        except ValueError:
            continue
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_rate_versus_POSITION_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()

def make_plots(raw_position_data, processed_position_data, spike_data=None, prm=None):
    plot_stops_on_track_offset(raw_position_data, processed_position_data, prm)
    criteria_plot_offset(processed_position_data, prm)
    plot_stops_on_track_offset_order(raw_position_data, processed_position_data, prm)
    plot_binned_velocity(raw_position_data, processed_position_data, prm, plot_beaconed=True, plot_non_beaconed=True)
    plot_binned_velocity(raw_position_data, processed_position_data, prm, plot_beaconed=False, plot_non_beaconed=True)
    plot_binned_velocity(raw_position_data, processed_position_data, prm, plot_beaconed=True, plot_non_beaconed=False)
    plot_binned_velocity(raw_position_data, processed_position_data, prm, plot_beaconed=True, plot_non_beaconed=True, ordered=True)
    plot_binned_velocity(raw_position_data, processed_position_data, prm, plot_beaconed=False, plot_non_beaconed=True, ordered=True)
    plot_binned_velocity(raw_position_data, processed_position_data, prm, plot_beaconed=True, plot_non_beaconed=False, ordered=True)
    #plot_stop_histogram(raw_position_data, processed_position_data, prm)
    #plot_speed_histogram(raw_position_data, processed_position_data, prm)

    if spike_data is not None:
        PostSorting.make_plots.plot_waveforms(spike_data, prm)
        PostSorting.make_plots.plot_spike_histogram(spike_data, prm)
        PostSorting.make_plots.plot_autocorrelograms(spike_data, prm)
        gc.collect()
        plot_spikes_on_track_cue_offset(spike_data, raw_position_data, processed_position_data, prm, prefix='_movement')
        plot_spikes_on_track_cue_offset_order(spike_data, raw_position_data, processed_position_data, prm, prefix='_movement')
        gc.collect()
        plot_convolved_rates_in_time(spike_data, prm)

        # plot_firing_rate_maps(spike_data, prm, prefix='_all')
        # plot_combined_spike_raster_and_rate(spike_data, raw_position_data, processed_position_data, prm, prefix='_all')
        # make_combined_figure(prm, spike_data, prefix='_all')


def criteria_plot_offset(processed_position_data, prm):

    print('I am plotting stop criteria with offset from the goal location...')
    save_path = prm.get_output_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6, 6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed, nonbeaconed, probe = split_stop_data_by_trial_type(processed_position_data)
    fs_beaconed, fs_nonbeaconed, fs_probe = split_stop_data_by_trial_type(processed_position_data, first_stops=True)

    plt.ylabel('Mean Stops', fontsize=12, labelpad=10)
    plt.xlabel('Location relative to Reward Zone (cm)', fontsize=12, labelpad=10)
    # plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(-200, 200)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot_cue_conditioned(ax, prm.get_track_length())
    x_max = max(processed_position_data.stop_trial_number) + 0.5
    plot_utility.style_vr_plot_offset(ax, x_max)
    #plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.12, right=0.87, top=0.92)

    beaconed_mean_stop = np.nanmean(beaconed[:,0])
    beaconed_std_stop = np.nanstd(beaconed[:,0])
    nonbeaconed_mean_stop = np.nanmean(nonbeaconed[:,0])
    nonbeaconed_std_stop = np.nanstd(nonbeaconed[:,0])

    fs_beaconed_mean_stop = np.nanmean(fs_beaconed[:, 0])
    fs_beaconed_std_stop = np.nanstd(fs_beaconed[:, 0])
    fs_nonbeaconed_mean_stop = np.nanmean(fs_nonbeaconed[:, 0])
    fs_nonbeaconed_std_stop = np.nanstd(fs_nonbeaconed[:, 0])

    plt.ylim(0, 3)
    plt.yticks(np.array((1, 2)), ("Non beaconed" , "Beaconed"))
    plt.errorbar(beaconed_mean_stop,       2.1, xerr=beaconed_std_stop, color="k", ecolor="k", fmt='o', capsize=0.2)
    plt.errorbar(nonbeaconed_mean_stop,    1.1, xerr=nonbeaconed_std_stop, color="r", ecolor="r", fmt='o', capsize=0.2)
    plt.errorbar(fs_beaconed_mean_stop,    1.9, xerr=fs_beaconed_std_stop, color="k", ecolor="k", fmt='^', capsize=0.2)
    plt.errorbar(fs_nonbeaconed_mean_stop, 0.9, xerr=fs_nonbeaconed_std_stop, color="r", ecolor="r", fmt='^', capsize=0.2)

    legend_elements = [Line2D([0], [0], marker='o', color='w', markeredgecolor="k", markerfacecolor='none', label='All stops'),
                       Line2D([0], [0], marker='^', color='w', markeredgecolor="k", markerfacecolor='none', label='First stops')]
    ax.legend(handles=legend_elements)
    ax.text(-160, 2.9, "25cm Threshold", fontsize=12)

    plt.plot(np.array([-25,-25]), np.array([0,3]), '--', color="k")
    plt.plot(np.array([ 25, 25]), np.array([0, 3]), '--', color="k")
    plt.tight_layout()

    plt.savefig(prm.get_output_path() + '/Figures/behaviour/stop_criteria' + '.png', dpi=200)
    plt.close()


def plot_stops_on_track_offset_order(raw_position_data, processed_position_data, prm):
    print('I am plotting stop rasta offset from the goal location...')
    save_path = prm.get_output_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed,nonbeaconed,probe = split_stop_data_by_trial_type(processed_position_data)

    #reward_locs = np.array(processed_position_data.rewarded_stop_locations)
    #reward_trials = np.array(processed_position_data.rewarded_trials)

    trial_bb_start, trial_bb_end = find_blackboxes_to_plot(raw_position_data, prm)

    # takes all plottables and reorders according to blackbox locations
    beaconed, nonbeaconed, probe, trial_bb_start, trial_bb_end = order_by_cue(beaconed, nonbeaconed, probe, trial_bb_start, trial_bb_end)

    fill_blackbox(trial_bb_start, ax)
    fill_blackbox(trial_bb_end, ax)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='0.5', markersize=2)
    ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='red', markersize=2)
    #ax.plot(probe[:,0], probe[:,1], 'o', color='blue', markersize=2)
    #ax.plot(reward_locs, reward_trials, '>', color='Red', markersize=3)
    plt.ylabel('Stops on trials (ordered)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    #plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(-200,200)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot_cue_conditioned(ax, prm.get_track_length())
    x_max = max(raw_position_data.trial_number) + 0.5
    plot_utility.style_vr_plot_offset(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_output_path() + '/Figures/behaviour/stop_raster_ordered' + '.png', dpi=200)
    plt.close()

def plot_binned_velocity(raw_position_data, processed_position_data, prm, plot_beaconed=True, plot_non_beaconed=True, ordered=False):
    print('I am plotting binned velocity offset from the goal location...')
    save_path = prm.get_output_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6, 6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    trial_bb_start, trial_bb_end = find_blackboxes_to_plot(raw_position_data, prm)
    n_beaconed_trials = int(processed_position_data.beaconed_total_trial_number[0])
    n_nonbeaconed_trials = int(processed_position_data.nonbeaconed_total_trial_number[0])

    if ordered:
        processed_position_data = PostSorting.vr_cued.order_by_goal_location(processed_position_data)
        _, _, _, trial_bb_start, trial_bb_end = order_by_cue(trial_bb_start=trial_bb_start, trial_bb_end=trial_bb_end)

    fill_blackbox(trial_bb_start, ax)
    fill_blackbox(trial_bb_end, ax)

    beaconed = list(processed_position_data.speed_trials_beaconed[:n_beaconed_trials])
    beaconed_trial_numbers = np.array(processed_position_data.speed_trials_beaconed_trial_number[:n_beaconed_trials])
    non_beaconed = list(processed_position_data.speed_trials_non_beaconed[:n_nonbeaconed_trials])
    non_beaconed_trial_numbers = np.array(processed_position_data.speed_trials_non_beaconed_trial_number[:n_nonbeaconed_trials])

    plot_utility.style_track_plot_cue_conditioned(ax, prm.get_track_length())
    x_max = max(raw_position_data.trial_number) + 0.5
    plot_utility.style_vr_plot_offset(ax, x_max)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.ylabel('Speeds on trials', fontsize=12, labelpad=10)
    plt.xlabel('Location relative to goal (cm)', fontsize=12, labelpad=10)
    # plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(-200, 200)

    max_speed = np.nanmax(np.concatenate(list(processed_position_data.speed_trials_binned[:int(max(processed_position_data.speed_trial_numbers))])).ravel())
    min_speed = np.nanmin(np.concatenate(list(processed_position_data.speed_trials_binned[:int(max(processed_position_data.speed_trial_numbers))])).ravel())

    # https://stackoverflow.com/questions/10533929/colors-of-rectangles-in-python
    normal = pl.Normalize(0, 20) # 0 to 25cm/s
    cax, _ = cbar.make_axes(ax)
    cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.jet, norm=normal)
    cax.set_ylabel('Speeds (cm/s)', fontsize=12, labelpad=10)

    if plot_beaconed:
        for i in range(len(beaconed)):
            goal_location = processed_position_data.goal_location_beaconed[i]
            bin_counter = 0.5
            for j in range(len(beaconed[i])):
                speed = normal(beaconed[i][j])
                if not math.isnan(speed):
                    ax.add_patch(plt.Rectangle((bin_counter-goal_location-0.5, beaconed_trial_numbers[i]-0.5), 1, 1, fc='r', color=pl.cm.jet(speed)))
                bin_counter+=1

    if plot_non_beaconed:
        for i in range(len(non_beaconed)):
            goal_location = processed_position_data.goal_location_non_beaconed[i]
            bin_counter = 0.5
            for j in range(len(non_beaconed[i])):
                speed = normal(non_beaconed[i][j])
                if not math.isnan(speed):
                    ax.add_patch(plt.Rectangle((bin_counter-goal_location-0.5, non_beaconed_trial_numbers[i]-0.5), 1, 1, fc='r', color=pl.cm.jet(speed)))
                bin_counter+=1

    # ax.plot(probe[:,0], probe[:,1], 'o', color='blue', markersize=2)
    # ax.plot(reward_locs, reward_trials, '>', color='Red', markersize=3)

    # plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.2, left=0.12, right=0.87, top=0.92)

    if plot_beaconed is True and plot_non_beaconed is not True:
        if not ordered:
            plt.savefig(prm.get_output_path() + '/Figures/behaviour/speed_on_trials_beaconed' + '.png', dpi=200)
        else:
            plt.savefig(prm.get_output_path() + '/Figures/behaviour/speed_on_trials_beaconed_ordered' + '.png', dpi=200)
    elif plot_beaconed is not True and plot_non_beaconed is True:
        if not ordered:
            plt.savefig(prm.get_output_path() + '/Figures/behaviour/speed_on_trials_nonbeaconed' + '.png', dpi=200)
        else:
            plt.savefig(prm.get_output_path() + '/Figures/behaviour/speed_on_trials_nonbeaconed_ordered' + '.png', dpi=200)
    elif plot_beaconed is True and plot_non_beaconed is True:
        if not ordered:
            plt.savefig(prm.get_output_path() + '/Figures/behaviour/speed_on_trials' + '.png', dpi=200)
        else:
            plt.savefig(prm.get_output_path() + '/Figures/behaviour/speed_on_trials_ordered' + '.png', dpi=200)
    plt.close()


def order_by_cue(beaconed=None, non_beaconed=None, probe=None, trial_bb_start=None, trial_bb_end=None):
    '''
    :param beaconed: 2d np array, one stop/spike per row [stop/spike location, trial number, trial type]
    :param non_beaconed: ''
    :param probe: ''
    :param trial_bb_start: list of black box centres relative to goal location per trial
    :param trial_bb_end: ''
    :return: complete set of reordered inputs
    '''
    tmp = np.array([np.arange(1, len(trial_bb_start)+1),trial_bb_start, trial_bb_end])
    sortedtmp = tmp[:, tmp[1].argsort()] # sorts by blackbox centres

    trial_bb_start = list(sortedtmp[1])
    trial_bb_end = list(sortedtmp[2])

    if beaconed is None:
        return beaconed, non_beaconed, probe, trial_bb_start, trial_bb_end

    sorted_trial_numbers = sortedtmp[0]
    new_trial_numbers = np.arange(1,len(trial_bb_start)+1)

    counter = 0
    for row in beaconed:
        if not math.isnan(row[1]):
            old_trial_number = int(row[1])
            new_trial_number = int(new_trial_numbers[sorted_trial_numbers == old_trial_number])
            row[1] = new_trial_number
            beaconed[counter] = row
        counter += 1

    counter = 0
    for row in non_beaconed:
        if not math.isnan(row[1]):
            old_trial_number = int(row[1])
            new_trial_number = int(new_trial_numbers[sorted_trial_numbers == old_trial_number])
            row[1] = new_trial_number
            non_beaconed[counter] = row
        counter += 1

    return beaconed, non_beaconed, probe, trial_bb_start, trial_bb_end


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()
    params.stop_threshold = 7.0
    params.track_length = 300
    params.cue_conditioned_goal = True
    params.set_output_path(r'C:\Users\44756\Desktop\test_recordings_waveform_matching')

    raw_position_data =       pd.read_pickle(r'C:\Users\44756\Desktop\test_recordings_waveform_matching\raw_position_data.pkl')  # m4
    processed_position_data = pd.read_pickle(r'C:\Users\44756\Desktop\test_recordings_waveform_matching\processed_position_data.pkl') #m4
    spatial_firing =          pd.read_pickle(r'C:\Users\44756\Desktop\test_recordings_waveform_matching\spatial_firing.pkl') #m4
    #m2 processed_position_data = pd.read_pickle('Z:\ActiveProjects\Harry\MouseVR\data\Cue_conditioned_cohort1_190902\M2_D21_2019-10-01_13-00-22\MountainSort\DataFrames\processed_position_data.pkl') #m2

    make_plots(raw_position_data, processed_position_data, spike_data=spatial_firing, prm=params)
    #criteria_plot_offset(processed_position_data, prm=params)
    #beaconed = np.array([[0.1, 1, 0],
    #                     [0.1, 2, 0],
    #                     [0.2, 2, 0],
    #                     [0.7, 3, 0],
    #                     [0.4, 3, 0],
    #                     [0.5, 4, 0]])
    #non_beaconed = np.array([[0.12, 5, 1],
    #                         [0.41, 7, 1],
    #                         [0.25, 7, 1],
    #                         [0.47, 8, 1],
    #                         [0.94, 9, 1],
    #                         [0.15, 9, 1]])

    #trial_bb_start = [0.1, 0.5, 0.11, 0.4, 0.2, 0.5, 0.1, 0.123, 0.421]
    #trial_bb_end = [0.4, 0.8, 0.41, 0.7, 0.5, 0.8, 0.4, 0.423, 0.721]
    #probe = 0
    #new_beaconed, new_non_beaconed, _, new_trial_bb_start, new_trial_bb_end = order_by_cue(beaconed, non_beaconed, probe, trial_bb_start, trial_bb_end)
    #print("hello there")



if __name__ == '__main__':
    main()
