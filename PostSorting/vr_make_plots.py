import os
import matplotlib.pylab as plt
import plot_utility
import numpy as np
import PostSorting.vr_stop_analysis
import PostSorting.vr_extract_data
import PostSorting.vr_cued_make_plots
import PostSorting.vr_spatial_data
from numpy import inf
import gc
from scipy import stats
import matplotlib.ticker as ticker
import pandas as pd
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import settings
from scipy import signal
import math
'''

# Plot basic info to check recording is good:
> movement channel
> trial channels (one and two)

'''

# plot the raw movement channel to check all is good
def plot_movement_channel(location, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(location)
    plt.savefig(save_path + '/movement' + '.png')
    plt.close()

# plot the trials to check all is good
def plot_trials(trials, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(trials)
    plt.savefig(save_path + '/trials' + '.png')
    plt.close()

def plot_velocity(velocity, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(velocity)
    plt.savefig(save_path + '/velocity' + '.png')
    plt.close()

def plot_running_mean_velocity(velocity, output_path):
    save_path = output_path + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(velocity)
    plt.savefig(save_path + '/running_mean_velocity' + '.png')
    plt.close()

# plot the raw trial channels to check all is good
def plot_trial_channels(trial1, trial2, output_path):
    plt.plot(trial1[0,:])
    plt.savefig(output_path + '/Figures/trial_type1.png')
    plt.close()
    plt.plot(trial2[0,:])
    plt.savefig(output_path + '/Figures/trial_type2.png')
    plt.close()


'''

# Plot behavioural info:
> stops on trials 
> avg stop histogram
> avg speed histogram
> combined plot

'''

def get_trial_color(trial_type):
    if trial_type == 0:
        return "black"
    elif trial_type == 1:
        return "red"
    elif trial_type == 2:
        return "blue"
    else:
        print("invalid trial-type passed to get_trial_color()")

def plot_stops_on_track(processed_position_data, output_path, track_length=200):
    print('I am plotting stop rasta...')
    save_path = output_path+'/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    for index, trial_row in processed_position_data.iterrows():
        trial_row = trial_row.to_frame().T.reset_index(drop=True)
        trial_type = trial_row["trial_type"].iloc[0]
        trial_number = trial_row["trial_number"].iloc[0]
        trial_stop_color = get_trial_color(trial_type)

        ax.plot(np.array(trial_row["stop_location_cm"].iloc[0]), trial_number*np.ones(len(trial_row["stop_location_cm"].iloc[0])), 'o', color=trial_stop_color, markersize=4)

    plt.ylabel('Stops on trials', fontsize=20, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
    plt.xlim(0,track_length)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)
    n_trials = len(processed_position_data)
    x_max = n_trials+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()


def plot_stop_histogram(processed_position_data, output_path, track_length=200):
    print('plotting stop histogram...')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)
    bin_size = 5

    beaconed_trials = processed_position_data[processed_position_data["trial_type"] == 0]
    non_beaconed_trials = processed_position_data[processed_position_data["trial_type"] == 1]
    probe_trials = processed_position_data[processed_position_data["trial_type"] == 2]

    beaconed_stops = plot_utility.pandas_collumn_to_numpy_array(beaconed_trials["stop_location_cm"])
    non_beaconed_stops = plot_utility.pandas_collumn_to_numpy_array(non_beaconed_trials["stop_location_cm"])
    probe_stops = plot_utility.pandas_collumn_to_numpy_array(probe_trials["stop_location_cm"])

    beaconed_stop_hist, bin_edges = np.histogram(beaconed_stops, bins=int(track_length/bin_size), range=(0, track_length))
    non_beaconed_stop_hist, bin_edges = np.histogram(non_beaconed_stops, bins=int(track_length/bin_size), range=(0, track_length))
    probe_stop_hist, bin_edges = np.histogram(probe_stops, bins=int(track_length/bin_size), range=(0, track_length))
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])

    ax.plot(bin_centres, beaconed_stop_hist/len(beaconed_trials), '-', color='Black')
    if len(non_beaconed_trials)>0:
        ax.plot(bin_centres, non_beaconed_stop_hist/len(non_beaconed_trials), '-', color='Red')
    if len(probe_trials)>0:
        ax.plot(bin_centres, probe_stop_hist/len(probe_trials), '-', color='Blue')

    plt.ylabel('Stops/Trial', fontsize=20, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
    plt.xlim(0,track_length)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)

    x_max = max(beaconed_stop_hist/len(beaconed_trials))+0.1
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/stop_histogram' + '.png', dpi=200)
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

def plot_speed_per_trial(processed_position_data, output_path, track_length=200):
    print('plotting speed heatmap...')
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    x_max = len(processed_position_data)
    fig = plt.figure(figsize=(4,(x_max/32)))
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    trial_speeds = plot_utility.pandas_collumn_to_2d_numpy_array(processed_position_data["speeds_binned_in_space"])
    where_are_NaNs = np.isnan(trial_speeds)
    trial_speeds[where_are_NaNs] = 0
    locations = np.arange(0, len(trial_speeds[0]))
    ordered = np.arange(0, len(trial_speeds), 1)
    X, Y = np.meshgrid(locations, ordered)
    cmap = plt.cm.get_cmap("jet")
    pcm = ax.pcolormesh(X, Y, trial_speeds, cmap=cmap, shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.mappable.set_clim(0, np.max(trial_speeds))
    plt.ylabel('Trial Number', fontsize=20, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
    plt.xlim(0,track_length)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.2, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/speed_heat_map' + '.png', dpi=200)
    plt.close()



def plot_speed_histogram(processed_position_data, output_path, track_length=200):
    save_path = output_path + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    speed_histogram = plt.figure(figsize=(6,4))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    for tt, c in zip([0,1,2], ["black", "red", "blue"]):
        tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
        if len(tt_processed_position_data)>0:
            trial_speeds = plot_utility.pandas_collumn_to_2d_numpy_array(tt_processed_position_data["speeds_binned_in_space"])
            trial_speeds_sem = stats.sem(trial_speeds, axis=0, nan_policy="omit")
            trial_speeds_avg = np.nanmean(trial_speeds, axis=0)
            bin_centres = np.array(processed_position_data["position_bin_centres"].iloc[0])
            ax.plot(bin_centres, trial_speeds_avg, color=c, linewidth=4)

    plt.ylabel('Speed (cm/s)', fontsize=25, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=25, labelpad = 10)
    plt.xlim(0,track_length)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, track_length)
    tick_spacing = 100
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plot_utility.style_vr_plot(ax, x_max=115)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(output_path + '/Figures/behaviour/speed_histogram' + '.png', dpi=200)
    plt.close()




def plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=200,
                         plot_trials=["beaconed", "non_beaconed", "probe"]):
    print('plotting spike rastas...')
    save_path = output_path + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = cluster_spike_data.firing_times.iloc[0]
        if len(firing_times_cluster)>1:

            x_max = len(processed_position_data)+1
            if x_max>100:
                spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
            else:
                spikes_on_track = plt.figure(figsize=(4,(x_max/20)))

            ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

            if "beaconed" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].beaconed_position_cm, cluster_spike_data.iloc[0].beaconed_trial_number, '|', color='Black', markersize=4)
            if "non_beaconed" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].nonbeaconed_position_cm, cluster_spike_data.iloc[0].nonbeaconed_trial_number, '|', color='Red', markersize=4)
            if "probe" in plot_trials:
                ax.plot(cluster_spike_data.iloc[0].probe_position_cm, cluster_spike_data.iloc[0].probe_trial_number, '|', color='Blue', markersize=4)

            plt.ylabel('Spikes on trials', fontsize=20, labelpad = 10)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
            plt.xlim(0,track_length)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            plot_utility.style_track_plot(ax, track_length)
            plot_utility.style_vr_plot(ax, x_max)
            plt.locator_params(axis = 'y', nbins  = 4)
            try:
                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            except ValueError:
                continue
            if len(plot_trials)<3:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + "_" + str("_".join(plot_trials)) + '.png', dpi=200)
            else:
                plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_track_firing_Cluster_' + str(cluster_id) + '.png', dpi=200)
            plt.close()


def plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=200):
    print('I am plotting firing rate maps...')
    save_path = output_path + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        if len(firing_times_cluster)>1:
            if "fr_binned_in_space" in list(cluster_spike_data):
                fr_column = "fr_binned_in_space"
            elif "fr_binned_in_space_smoothed" in list(cluster_spike_data):
                fr_column = "fr_binned_in_space_smoothed"
            fr_binned_in_space = np.array(cluster_spike_data[fr_column].iloc[0])
            fr_binned_in_space_bin_centres = np.array(cluster_spike_data['fr_binned_in_space_bin_centres'].iloc[0])[0]

            spikes_on_track = plt.figure()
            spikes_on_track.set_size_inches(5, 5, forward=True)
            ax = spikes_on_track.add_subplot(1, 1, 1)
            plot_utility.style_track_plot(ax, track_length)
            y_max=0

            for tt, c in zip([0, 1, 2], ["black", "red", "blue"]):
                tt_processed_position_data = processed_position_data[processed_position_data["trial_type"] == tt]
                tt_trial_numbers = np.asarray(tt_processed_position_data["trial_number"])
                tt_fr_binned_in_space = fr_binned_in_space[tt_trial_numbers-1]
                ax.fill_between(fr_binned_in_space_bin_centres, np.nanmean(tt_fr_binned_in_space, axis=0)-stats.sem(tt_fr_binned_in_space, axis=0), np.nanmean(tt_fr_binned_in_space, axis=0)+stats.sem(tt_fr_binned_in_space, axis=0), color=c, alpha=0.3)
                ax.plot(fr_binned_in_space_bin_centres, np.nanmean(tt_fr_binned_in_space, axis=0), color=c)

                fr_max = max(np.nanmean(tt_fr_binned_in_space, axis=0)+stats.sem(tt_fr_binned_in_space, axis=0))
                y_max = max([y_max, fr_max])
                y_max = np.ceil(y_max)

            plt.ylabel('Firing Rate (Hz)', fontsize=20, labelpad = 20)
            plt.xlabel('Location (cm)', fontsize=20, labelpad = 20)
            plt.xlim(0, track_length)
            tick_spacing = 100
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plot_utility.style_vr_plot(ax, x_max=y_max)
            ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])
            plt.locator_params(axis = 'y', nbins  = 4)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.tight_layout()
            plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_rate_map_Cluster_' + str(cluster_id) + '.png', dpi=300)
            plt.close()

    return spike_data

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
        plt.ylabel('Firing rate (Hz)', fontsize=20, labelpad = 10)
        plt.xlabel('Speed (cm/sec)', fontsize=20, labelpad = 10)
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
        plt.ylabel('Firing rate (Hz)', fontsize=20, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=20, labelpad = 10)
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

def make_plots(processed_position_data, spike_data, output_path, track_length=settings.track_length):
    # Create plots for the VR experiments
    
    plot_stops_on_track(processed_position_data, output_path, track_length=track_length)
    plot_stop_histogram(processed_position_data, output_path, track_length=track_length)
    plot_speed_histogram(processed_position_data, output_path, track_length=track_length)
    plot_speed_per_trial(processed_position_data, output_path, track_length=track_length)

    if spike_data is not None:
        PostSorting.make_plots.plot_waveforms(spike_data, output_path)
        PostSorting.make_plots.plot_spike_histogram(spike_data, output_path)
        PostSorting.make_plots.plot_autocorrelograms(spike_data, output_path)
        gc.collect()
        plot_firing_rate_maps(spike_data, processed_position_data, output_path, track_length=track_length)
        plot_spikes_on_track(spike_data, processed_position_data, output_path, track_length=track_length,
                             plot_trials=["beaconed", "non_beaconed", "probe"])



#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


if __name__ == '__main__':
    main()


