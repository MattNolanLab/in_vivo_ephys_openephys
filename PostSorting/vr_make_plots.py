import os
import matplotlib.pylab as plt
import plot_utility
import numpy as np
import PostSorting.vr_stop_analysis
import PostSorting.vr_extract_data
from numpy import inf
import gc
import math
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

'''

# Plot basic info to check recording is good:
> movement channel
> trial channels (one and two)

'''

# plot the raw movement channel to check all is good
def plot_movement_channel(location, prm):
    save_path = prm.get_output_path() + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(location)
    plt.savefig(save_path + '/movement' + '.png')
    plt.close()

# plot the trials to check all is good
def plot_trials(trials, prm):
    plt.plot(trials)
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/trials' + '.png')
    plt.close()

# plot the raw trial channels to check all is good
def plot_trial_channels(trial1, trial2, prm):
    plt.plot(trial1[0,:])
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/trial_type1' + '.png')
    plt.close()
    plt.plot(trial2[0,:])
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/trial_type2' + '.png')
    plt.close()


'''

# Plot behavioural info:
> stops on trials 
> avg stop histogram
> avg speed histogram
> combined plot

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


def plot_stops_on_track(raw_position_data, processed_position_data, prm):
    print('I am plotting stop rasta...')
    save_path = prm.get_output_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed,nonbeaconed,probe = split_stop_data_by_trial_type(processed_position_data)
    reward_locs = np.array(processed_position_data.rewarded_stop_locations)
    reward_trials = np.array(processed_position_data.rewarded_trials)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='0.5', markersize=2)
    ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='red', markersize=2)
    ax.plot(probe[:,0], probe[:,1], 'o', color='blue', markersize=2)
    ax.plot(reward_locs, reward_trials, '>', color='Red', markersize=3)
    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    #plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(raw_position_data.trial_number)+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_output_path() + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()

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


def plot_combined_behaviour(raw_position_data,processed_position_data, prm):
    print('making combined behaviour plot...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    combined = plt.figure(figsize=(6,9))
    ax = combined.add_subplot(3, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed,nonbeaconed,probe = split_stop_data_by_trial_type(processed_position_data)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='0.5', markersize=2)
    ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='red', markersize=2)
    ax.plot(probe[:,0], probe[:,1], 'o', color='blue', markersize=2)
    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(raw_position_data.trial_number)+0.5
    plot_utility.style_vr_plot(ax, x_max)

    ax = combined.add_subplot(3, 1, 2)  # specify (nrows, ncols, axnum)
    position_bins = np.array(processed_position_data["position_bins"].dropna(axis=0))
    average_stops = np.array(processed_position_data["average_stops"].dropna(axis=0))
    ax.plot(position_bins,average_stops, '-', color='Black')
    plt.ylabel('Stops (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(processed_position_data.average_stops)+0.5
    plot_utility.style_vr_plot(ax, x_max)

    ax = combined.add_subplot(3, 1, 3)  # specify (nrows, ncols, axnum)
    position_bins = np.array(processed_position_data["position_bins"].dropna(axis=0))
    average_speed = np.array(processed_position_data["binned_speed_ms"].dropna(axis=0))
    ax.plot(position_bins,average_speed, '-', color='Black')
    plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(processed_position_data.binned_speed_ms)+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .3, wspace = .3,  bottom = 0.06, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/behaviour/combined_behaviour' + '.png', dpi=200)
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

        # uncomment if you want to plot stops
        # ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)
        # ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)
        # ax.plot(probe[:,0], probe[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)

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

        plot_utility.style_track_plot_cue_conditioned(ax, 300)
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
                             np.zeros(len(spike_data.loc[cluster_index].beaconed_trial_number))])

        nonbeaconed = np.array([spike_data.loc[cluster_index].nonbeaconed_position_cm,
                                 spike_data.loc[cluster_index].nonbeaconed_trial_number,
                                 np.zeros(len(spike_data.loc[cluster_index].nonbeaconed_trial_number))])

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

        plot_utility.style_track_plot_cue_conditioned(ax, 300)
        plot_utility.style_vr_plot_offset(ax, x_max)
        plt.locator_params(axis='y', nbins=4)
        try:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        except ValueError:
            continue
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_track_firing_orded_Cluster_' + str(cluster_index + 1) + '.png', dpi=200)
        plt.close()

def plot_spikes_on_track(spike_data,raw_position_data,processed_position_data, prm, prefix):
    print('plotting spike rastas...')
    save_path = prm.get_output_path() + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    rewarded_locations = np.array(processed_position_data['rewarded_stop_locations'].dropna(axis=0)) #
    rewarded_trials = np.array(processed_position_data['rewarded_trials'].dropna(axis=0))

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        x_max = max(np.array(spike_data.at[cluster_index, 'beaconed_trial_number'])) + 1
        spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        #uncomment if you want to plot stops
        #ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)
        #ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)
        #ax.plot(probe[:,0], probe[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)

        ax.plot(spike_data.loc[cluster_index].beaconed_position_cm, spike_data.loc[cluster_index].beaconed_trial_number, '|', color='Black', markersize=4)
        ax.plot(spike_data.loc[cluster_index].nonbeaconed_position_cm, spike_data.loc[cluster_index].nonbeaconed_trial_number, '|', color='Red', markersize=4)
        ax.plot(spike_data.loc[cluster_index].probe_position_cm, spike_data.loc[cluster_index].probe_trial_number, '|', color='Blue', markersize=4)
        ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=3)

        plt.ylabel('Spikes on trials', fontsize=12, labelpad = 10)
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
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_track_firing_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
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
    if prm.cue_conditioned_goal is False:
        plot_stops_on_track(raw_position_data, processed_position_data, prm)
        plot_stop_histogram(raw_position_data, processed_position_data, prm)
        plot_speed_histogram(raw_position_data, processed_position_data, prm)
        if spike_data is not None:
            PostSorting.make_plots.plot_waveforms(spike_data, prm)
            PostSorting.make_plots.plot_spike_histogram(spike_data, prm)
            PostSorting.make_plots.plot_autocorrelograms(spike_data, prm)
            gc.collect()
            plot_firing_rate_maps(spike_data, prm, prefix='_all')
            #plot_convolved_rates_in_time(spike_data, prm)
            #plot_combined_spike_raster_and_rate(spike_data, raw_position_data, processed_position_data, prm, prefix='_all')
            #make_combined_figure(prm, spike_data, prefix='_all')
    else:
        plot_stops_on_track_offset(raw_position_data, processed_position_data, prm)
        criteria_plot_offset(processed_position_data, prm)
        plot_stops_on_track_offset_order(raw_position_data, processed_position_data, prm)
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


# unused code but might use in future

'''
def plot_combined_spike_raster_and_rate(spike_data,raw_position_data,processed_position_data, prm, prefix):
    print('plotting combined spike rastas and spike rate...')
    save_path = prm.get_output_path() + '/Figures/combined_spike_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    rewarded_locations = np.array(processed_position_data['rewarded_stop_locations'].dropna(axis=0))
    rewarded_trials = np.array(processed_position_data['rewarded_trials'].dropna(axis=0))

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        spikes_on_track = plt.figure(figsize=(6,10))

        ax = spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
        cluster_firing_indices = spike_data.firing_times[cluster_index]
        ax.plot(spike_data.loc[cluster_index].beaconed_position_cm, spike_data.loc[cluster_index].beaconed_trial_number, '|', color='Black', markersize=4)
        ax.plot(spike_data.loc[cluster_index].nonbeaconed_position_cm, spike_data.loc[cluster_index].nonbeaconed_trial_number, '|', color='Red', markersize=4)
        ax.plot(spike_data.loc[cluster_index].probe_position_cm, spike_data.loc[cluster_index].probe_trial_number, '|', color='Blue', markersize=4)
        ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=2)

        plt.ylabel('Spikes on trials', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plot_utility.style_track_plot(ax, 200)
        x_max = max(raw_position_data.trial_number[cluster_firing_indices])+0.5
        plot_utility.style_vr_plot(ax, x_max)

        ax = spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
        unsmooth_b = np.array(spike_data.at[cluster_index, 'avg_b_spike_rate'])
        unsmooth_nb = np.array(spike_data.at[cluster_index, 'avg_nb_spike_rate'])
        unsmooth_p = np.array(spike_data.at[cluster_index, 'avg_p_spike_rate'])
        ax.plot(unsmooth_b, '-', color='Black')
        ax.plot(unsmooth_nb, '-', color='Red')
        ax.plot(unsmooth_p, '-', color='Blue')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        nb_x_max = np.nanmax(np.array(spike_data.at[cluster_index, 'avg_b_spike_rate']))
        b_x_max = np.nanmax(np.array(spike_data.at[cluster_index, 'avg_nb_spike_rate']))
        p_x_max = np.nanmax(np.array(spike_data.at[cluster_index, 'avg_p_spike_rate']))
        if b_x_max > nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, b_x_max)
        elif b_x_max < nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, nb_x_max)
        elif b_x_max > nb_x_max and b_x_max < p_x_max:
            plot_utility.style_vr_plot(ax, p_x_max)
        plot_utility.style_track_plot(ax, 200)
        plt.subplots_adjust(hspace = .2, wspace = .2,  bottom = 0.06, left = 0.12, right = 0.87, top = 0.92)
        plt.savefig(prm.get_output_path() + '/Figures/combined_spike_plots/' + spike_data.session_id[cluster_index] + '_track_firing_Cluster_' + str(cluster_index +1) + str(prefix) + '.png', dpi=200)
        plt.close()


def make_combined_figure(prm, spatial_firing, prefix):
    print('I will make the combined images now.')
    save_path = prm.get_output_path() + '/Figures/combined'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.close('all')
    figures_path = prm.get_output_path() + '/Figures/'
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1

        spike_raster_path = figures_path + 'combined_spike_plots/' + spatial_firing.session_id[cluster] + '_track_firing_Cluster_' + str(cluster +1) + str(prefix) + '.png'
        spike_histogram_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_histogram.png'
        autocorrelogram_10_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_autocorrelogram_10ms.png'
        autocorrelogram_250_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_autocorrelogram_250ms.png'
        waveforms_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_waveforms.png'
        combined_behaviour_path = figures_path + 'behaviour/combined_behaviour.png'
        grid = plt.GridSpec(6, 3, wspace=0.003, hspace=0.01)
        if os.path.exists(waveforms_path):
            waveforms = mpimg.imread(waveforms_path)
            waveforms_plot = plt.subplot(grid[0, 0])
            waveforms_plot.axis('off')
            waveforms_plot.imshow(waveforms)
        if os.path.exists(spike_histogram_path):
            spike_hist = mpimg.imread(spike_histogram_path)
            spike_hist_plot = plt.subplot(grid[1, 0])
            spike_hist_plot.axis('off')
            spike_hist_plot.imshow(spike_hist)
        if os.path.exists(autocorrelogram_10_path):
            autocorrelogram_10 = mpimg.imread(autocorrelogram_10_path)
            autocorrelogram_10_plot = plt.subplot(grid[2, 0])
            autocorrelogram_10_plot.axis('off')
            autocorrelogram_10_plot.imshow(autocorrelogram_10)
        if os.path.exists(autocorrelogram_250_path):
            autocorrelogram_250 = mpimg.imread(autocorrelogram_250_path)
            autocorrelogram_250_plot = plt.subplot(grid[3, 0])
            autocorrelogram_250_plot.axis('off')
            autocorrelogram_250_plot.imshow(autocorrelogram_250)
        if os.path.exists(spike_raster_path):
            spike_raster = mpimg.imread(spike_raster_path)
            spike_raster_plot = plt.subplot(grid[:, 1])
            spike_raster_plot.axis('off')
            spike_raster_plot.imshow(spike_raster)
        if os.path.exists(combined_behaviour_path):
            combined_behaviour = mpimg.imread(combined_behaviour_path)
            combined_behaviour_plot = plt.subplot(grid[:, 2])
            combined_behaviour_plot.axis('off')
            combined_behaviour_plot.imshow(combined_behaviour)
        plt.subplots_adjust(hspace = .0, wspace = .0,  bottom = 0.06, left = 0.06, right = 0.94, top = 0.94)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + str(prefix) + '.png', dpi=1000)
        plt.close()

'''
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
    beaconed_std_stop = np.nanstd(beaconed[:,1])
    nonbeaconed_mean_stop = np.nanmean(nonbeaconed[:,0])
    nonbeaconed_std_stop = np.nanstd(nonbeaconed[:,1])

    fs_beaconed_mean_stop = np.nanmean(fs_beaconed[:, 0])
    fs_beaconed_std_stop = np.nanstd(fs_beaconed[:, 1])
    fs_nonbeaconed_mean_stop = np.nanmean(fs_nonbeaconed[:, 0])
    fs_nonbeaconed_std_stop = np.nanstd(fs_nonbeaconed[:, 1])

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

def order_by_cue(beaconed, non_beaconed, probe, trial_bb_start, trial_bb_end):
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

    sorted_trial_numbers = sortedtmp[0]
    new_trial_numbers = np.arange(1,len(trial_bb_start)+1)

    counter = 0
    for row in beaconed:
        if not math.isnan(row[1]):
            old_trial_number = int(row[1])
            new_trial_number = int(sorted_trial_numbers[new_trial_numbers == old_trial_number])
            row[1] = new_trial_number
            beaconed[counter] = row
        counter += 1

    counter = 0
    for row in non_beaconed:
        if not math.isnan(row[1]):
            old_trial_number = int(row[1])
            new_trial_number = int(sorted_trial_numbers[new_trial_numbers == old_trial_number])
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
