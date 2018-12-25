import os
import matplotlib.pylab as plt
import plot_utility
import numpy as np
import PostSorting.vr_stop_analysis
import matplotlib.image as mpimg
import pandas as pd
from scipy import stats



def plot_spikes_on_track(recording_folder,spike_data,processed_position_data, prm, prefix):
    print('plotting spike rasters...')
    save_path = recording_folder + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        x_max = max(np.array(spike_data.at[cluster_index, 'beaconed_trial_number']))+1
        spikes_on_track = plt.figure(figsize=(4,(x_max/35)))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        ax.plot(spike_data.loc[cluster_index].beaconed_position_cm, spike_data.loc[cluster_index].beaconed_trial_number, '|', color='Black', markersize=3)
        ax.plot(spike_data.loc[cluster_index].nonbeaconed_position_cm, spike_data.loc[cluster_index].nonbeaconed_trial_number, '|', color='Red', markersize=3)
        ax.plot(spike_data.loc[cluster_index].probe_position_cm, spike_data.loc[cluster_index].probe_trial_number, '|', color='Blue', markersize=3)

        plt.ylabel('Spikes on trials', fontsize=12, labelpad = 8)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 7)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plot_utility.style_track_plot(ax, 200)
        plot_utility.style_vr_plot(ax, x_max)
        plt.locator_params(axis = 'y', nbins  = 4)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(recording_folder + '/Figures/spike_trajectories/' + spike_data.session_id[cluster_index] + '_track_firing_Cluster_' + str(cluster_index +1) + str(prefix) + '.png', dpi=200)
        plt.close()


def find_max_y_value(spike_data, cluster_index):
    nb_x_max = np.nanmax(np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_b']))
    b_x_max = np.nanmax(np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_nb']))
    p_x_max = np.nanmax(np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_p']))
    if b_x_max > nb_x_max and b_x_max > p_x_max:
        x_max = b_x_max
    elif nb_x_max > b_x_max and nb_x_max > p_x_max:
        x_max = nb_x_max
    elif p_x_max > b_x_max and p_x_max > nb_x_max:
        x_max = p_x_max

    x_max = x_max+(x_max/10)
    return x_max


def plot_firing_rate_maps(recording_folder, spike_data, prefix):
    print('I am plotting firing rate maps...')
    save_path = recording_folder + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        avg_spikes_on_track = plt.figure(figsize=(4,3))

        unsmooth_b = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_b'])
        unsmooth_nb = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_nb'])
        unsmooth_p = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_p'])

        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(unsmooth_b, '-', color='Black')
        ax.plot(unsmooth_nb, '-', color='Red')
        ax.plot(unsmooth_p, '-', color='Blue')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 8)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 7)
        plt.xlim(0,200)

        x_max = find_max_y_value(spike_data, cluster_index)
        plt.locator_params(axis = 'y', nbins  = 4)
        plot_utility.style_vr_plot(ax, x_max)
        plot_utility.style_track_plot(ax, 200)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.savefig(recording_folder + '/Figures/spike_rate/' + spike_data.session_id[cluster_index] + '_rate_map_Cluster_' + str(cluster_index +1) + str(prefix) + '.png', dpi=200)
        plt.close()


def get_spike_interval(cluster_index,spike_data):
    firing_times = np.array(spike_data.at[cluster_index, 'firing_times'])
    trial_numbers = np.array(spike_data.at[cluster_index, 'trial_numbers'])
    for tcount, trial in enumerate(np.unique(trial_numbers)):
        spikes_on_trial = np.take(firing_times, np.where(trial_numbers == trial)[0])




def plot_spikes_aligned_by_time(recording_folder,spike_data,processed_position_data, prm, prefix):
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        get_spike_interval(cluster_index,spike_data)


