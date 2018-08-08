import cmocean
import matplotlib.pylab as plt
import os
import plot_utility
import math
import numpy as np
import random

from scipy.interpolate import spline
import PostSorting.parameters

import pandas as pd
import PostSorting.open_field_firing_fields


def plot_position(position_data):
    plt.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=5)
    plt.close()


def plot_spikes_on_trajectory(position_data, spike_data, prm):

    cluster_id = 5  # this is just a test plot, it plots cluster 5
    spikes_on_track = plt.figure()
    ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1, alpha=0.7)
    ax.scatter(spike_data.position_x[cluster_id], spike_data.position_y[cluster_id], color='red', marker='o', s=10, zorder=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        right=False,
        left=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    ax.set_aspect('equal')

    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/spatial_firing.png')
    plt.close()


def plot_coverage(position_heat_map, prm):
    coverage = plt.figure()
    ax = coverage.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.imshow(position_heat_map, cmap='jet', interpolation='nearest')
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/heatmap.png')
    plt.close()


def plot_firing_rate_maps(spatial_firing, prm):
    for cluster in range(len(spatial_firing)):
        firing_rate_map = spatial_firing.firing_maps[cluster]
       #  plt.imshow(firing_rate_map, cmap='jet', interpolation='nearest')
        firing_rate_map_fig = plt.figure()
        ax = firing_rate_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.imshow(firing_rate_map, cmap='jet', interpolation='nearest')
        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/' + spatial_firing.session_id[cluster] + 'rate_map_' + str(cluster + 1) + '.png')
        plt.close()


def plot_hd(spatial_firing, position_data, prm):
    save_path = prm.get_local_recording_folder_path() + '/Figures/head_direction_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        x_positions = spatial_firing.position_x[cluster]
        y_positions = spatial_firing.position_y[cluster]
        hd = spatial_firing.hd[cluster]
        hd_map_fig = plt.figure()
        ax = hd_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax = plot_utility.style_open_field_plot(ax)
        ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1,
                alpha=0.2)
        hd_plot = ax.scatter(x_positions, y_positions, s=20, c=hd, vmin=-180, vmax=180, marker='o', cmap=cmocean.cm.phase)
        plt.colorbar(hd_plot)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_map_' + str(cluster + 1) + '.png')
        plt.close()


def plot_polar_head_direction_histogram(hd_hist, spatial_firing, prm):
    print('I will make the polar HD plots now.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/head_direction_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        hd_hist_cluster = spatial_firing.hd_spike_histogram[cluster]
        theta = np.linspace(0, 2*np.pi, 361)  # x axis
        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        ax.plot(theta[:-1], hd_hist_cluster, color='red', linewidth=2)
        ax.plot(theta[:-1], hd_hist*(max(hd_hist_cluster)/max(hd_hist)), color='black', linewidth=2)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_polar_' + str(cluster + 1) + '.png')
        plt.close()


def plot_hd_for_firing_fields(spatial_firing, prm):
    print('I will make the polar HD plots for individual firing fields now.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/firing_field_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        firing_rate_map = spatial_firing.firing_maps[cluster]
        firing_rate_map_fig = plt.figure()
        ax = firing_rate_map_fig.add_subplot(1, 1, 1)
        ax.imshow(firing_rate_map)

        if len(spatial_firing.firing_fields[cluster]) > 0:
            firing_fields_cluster = spatial_firing.firing_fields[cluster]
            colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(firing_fields_cluster))]
            for field_id, field in enumerate(firing_fields_cluster):
                for bin in field:
                    plt.scatter(bin[1], bin[0], color=colors[field_id], marker='o', s=20)


def main():
    prm = PostSorting.parameters.Parameters()
    prm.set_local_recording_folder_path('C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of/')
    firing_rate_maps = np.load('C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of/M5_2018-03-06_15-34-44_of.npy')
    cluster_id = np.arange(len(firing_rate_maps))
    spatial_firing = pd.DataFrame(cluster_id)
    spatial_firing['firing_maps'] = list(firing_rate_maps)
    spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing)
    plot_hd_for_firing_fields(spatial_firing, prm)
if __name__ == '__main__':
    main()
