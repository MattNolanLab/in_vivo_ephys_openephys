import cmocean
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import os
import plot_utility
import math
import numpy as np
import PostSorting.parameters
import  PostSorting.open_field_head_direction

import pandas as pd
import PostSorting.open_field_firing_fields


def plot_position(position_data):
    plt.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=5)
    plt.close()


def plot_spikes_on_trajectory(position_data, spike_data, prm):
    print('I will make scatter plots of spikes on the trajectory of the animal.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/firing_scatters'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_id in range(len(spike_data)):
        spikes_on_track = plt.figure()
        spikes_on_track.set_size_inches(5, 5, forward=True)
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
            labelbottom=False)  # labels along the bottom edge are off
        ax.set_aspect('equal')
        plt.title('spikes on trajectory', y=1.08)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_id] + '_' + str(cluster_id + 1) + '_spikes_on_trajectory.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_coverage(position_heat_map, prm):
    print('I will plot a heat map of the position of the animal to show coverage.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/session'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    coverage = plt.figure()
    coverage.set_size_inches(5, 5, forward=True)
    ax = coverage.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.imshow(position_heat_map, cmap=cmocean.cm.thermal, interpolation='nearest')
    plt.title('coverage', y=1.08)
    plt.savefig(save_path + '/heatmap.png', dpi=300)
    plt.close()


def plot_firing_rate_maps(spatial_firing, prm):
    print('I will make rate map plots.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/rate_maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        firing_rate_map = spatial_firing.firing_maps[cluster]
        firing_rate_map_fig = plt.figure()
        firing_rate_map_fig.set_size_inches(5, 5, forward=True)
        ax = firing_rate_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.imshow(firing_rate_map, cmap='jet', interpolation='nearest')
        plt.title('max fr: ' + str(round(spatial_firing.max_firing_rate[cluster], 2)) + ' Hz', y=1.08)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_rate_map_' + str(cluster + 1) + '.png', dpi=300)
        plt.close()


def plot_hd(spatial_firing, position_data, prm):
    print('I will plot HD on open field maps as a scatter plot for each cluster.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/head_direction_plots_2d'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        x_positions = spatial_firing.position_x[cluster]
        y_positions = spatial_firing.position_y[cluster]
        hd = spatial_firing.hd[cluster]
        hd_map_fig = plt.figure()
        hd_map_fig.set_size_inches(5, 5, forward=True)
        ax = hd_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax = plot_utility.style_open_field_plot(ax)
        ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1,
                alpha=0.2)
        hd_plot = ax.scatter(x_positions, y_positions, s=20, c=hd, vmin=-180, vmax=180, marker='o', cmap=cmocean.cm.phase)
        plt.colorbar(hd_plot, fraction=0.046, pad=0.04)
        plt.title('head-direction', y=1.08)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_map_' + str(cluster + 1) + '.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_polar_head_direction_histogram(hd_hist, spatial_firing, prm):
    print('I will make the polar HD plots now.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/head_direction_plots_polar'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        hd_polar_fig = plt.figure()
        hd_polar_fig.set_size_inches(5, 5, forward=True)
        ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        hd_hist_cluster = spatial_firing.hd_spike_histogram[cluster]
        theta = np.linspace(0, 2*np.pi, 361)  # x axis
        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        ax.plot(theta[:-1], hd_hist_cluster, color='red', linewidth=2)
        ax.plot(theta[:-1], hd_hist*(max(hd_hist_cluster)/max(hd_hist)), color='black', linewidth=2)
        plt.title('max fr: ' + str(round(spatial_firing.max_firing_rate_hd[cluster], 2)) + ' Hz' + ', preferred HD: ' + str(round(spatial_firing.preferred_HD[cluster][0], 0)), y=1.08)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_polar_' + str(cluster + 1) + '.png', dpi=300)
        plt.close()


def mark_firing_field_with_scatter(field, plot, colors, field_id):
    for bin in field:
        plot.scatter(bin[1], bin[0], color=colors[field_id], marker='o', s=5)
    return plot


# generate more random colors if necessary
def generate_colors(number_of_firing_fields):
    colors = [[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]]  # green, yellow, cyan, pink
    if number_of_firing_fields > len(colors):
        for i in range(number_of_firing_fields):
            colors.append(plot_utility.generate_new_color(colors, pastel_factor=0.9))
    return colors


def plot_hd_for_firing_fields(spatial_firing, spatial_data, prm):
    print('I will make the polar HD plots for individual firing fields now.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/firing_field_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        number_of_firing_fields = len(spatial_firing.firing_fields[cluster])
        firing_rate_map = spatial_firing.firing_maps[cluster]
        if number_of_firing_fields > 0:
            plt.clf()
            fig = plt.figure()
            fig.set_size_inches(20, 10, forward=True)
            gs = GridSpec(2, number_of_firing_fields + 1)
            of_plot = plt.subplot(gs[0:2, 0:2])
            of_plot.axis('off')
            of_plot.imshow(firing_rate_map)

            firing_fields_cluster = spatial_firing.firing_fields[cluster]
            colors = generate_colors(number_of_firing_fields)

            for field_id, field in enumerate(firing_fields_cluster):
                of_plot = mark_firing_field_with_scatter(field, of_plot, colors, field_id)

                hd_in_field_session = PostSorting.open_field_head_direction.get_hd_in_firing_rate_bins_for_session(spatial_data, field, prm)
                hd_in_field_cluster = PostSorting.open_field_head_direction.get_hd_in_firing_rate_bins_for_cluster(spatial_firing, field, cluster, prm)
                hd_hist_session = PostSorting.open_field_head_direction.get_hd_histogram(hd_in_field_session)
                hd_hist_session /= prm.get_sampling_rate()
                hd_hist_cluster = PostSorting.open_field_head_direction.get_hd_histogram(hd_in_field_cluster)
                hd_hist_cluster = np.divide(hd_hist_cluster, hd_hist_session, out=np.zeros_like(hd_hist_cluster), where=hd_hist_session != 0)

                theta = np.linspace(0, 2*np.pi, 361)  # x axis
                plot_row = field_id % 2
                # print(plot_row)
                plot_col = math.floor(field_id/2) + 1
                # print(plot_col)
                hd_plot_field = plt.subplot(gs[plot_row, plot_col], polar=True)
                hd_plot_field = plot_utility.style_polar_plot(hd_plot_field)

                hd_plot_field.plot(theta[:-1], hd_hist_session*(max(hd_hist_cluster)/max(hd_hist_session)), color='black', linewidth=2, alpha=0.9)
                hd_plot_field.plot(theta[:-1], hd_hist_cluster, color=colors[field_id], linewidth=2)

            #plt.tight_layout()
            gs.update(wspace=1, hspace=0.5)
            plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_firing_fields_' + str(cluster + 1) + '.png', dpi=300)
            plt.close()


def make_combined_figure(prm, spatial_firing):
    print('I will make the combined images now.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/combined'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.close('all')
    figures_path = prm.get_local_recording_folder_path() + '/Figures/'
    for cluster in range(len(spatial_firing)):
        coverage_path = figures_path + 'session/heatmap.png'
        spike_scatter_path = figures_path + 'firing_scatters/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spikes_on_trajectory.png'
        rate_map_path = figures_path + 'rate_maps/' + spatial_firing.session_id[cluster] + '_rate_map_' + str(cluster + 1) + '.png'
        head_direction_polar_path = figures_path + 'head_direction_plots_polar/' + spatial_firing.session_id[cluster] + '_hd_polar_' + str(cluster + 1) + '.png'
        head_direction_map_path = figures_path + 'head_direction_plots_2d/' + spatial_firing.session_id[cluster] + '_hd_map_' + str(cluster + 1) + '.png'
        firing_fields_path = figures_path + 'firing_field_plots/' + spatial_firing.session_id[cluster] + '_firing_fields_' + str(cluster + 1) + '.png'
        spike_histogram_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_hitogram.png'

        grid = plt.GridSpec(5, 3, wspace=0, hspace=0)
        if os.path.exists(spike_scatter_path):
            spike_hist = mpimg.imread(spike_histogram_path)
            spike_hist_plot = plt.subplot(grid[0, 0])
            spike_hist_plot.axis('off')
            spike_hist_plot.imshow(spike_hist)

        if os.path.exists(spike_scatter_path):
            spike_scatter = mpimg.imread(spike_scatter_path)
            spike_scatter_plot = plt.subplot(grid[1, 0])
            spike_scatter_plot.axis('off')
            spike_scatter_plot.imshow(spike_scatter)
        if os.path.exists(rate_map_path):
            rate_map = mpimg.imread(rate_map_path)
            rate_map_plot = plt.subplot(grid[1, 1])
            rate_map_plot.axis('off')
            rate_map_plot.imshow(rate_map)
        if os.path.exists(coverage_path):
            coverage = mpimg.imread(coverage_path)
            coverage_plot = plt.subplot(grid[1, 2])
            coverage_plot.axis('off')
            coverage_plot.imshow(coverage)
        if os.path.exists(head_direction_polar_path):
            polar_hd = mpimg.imread(head_direction_polar_path)
            polar_hd_plot = plt.subplot(grid[2, 0])
            polar_hd_plot.axis('off')
            polar_hd_plot.imshow(polar_hd)
        if os.path.exists(head_direction_map_path):
            hd_map = mpimg.imread(head_direction_map_path)
            hd_map_plot = plt.subplot(grid[2, 1])
            hd_map_plot.axis('off')
            hd_map_plot.imshow(hd_map)
        if os.path.exists(firing_fields_path):
            firing_fields = mpimg.imread(firing_fields_path)
            firing_fields_plot = plt.subplot(grid[3:5, 0:2])
            firing_fields_plot.axis('off')
            firing_fields_plot.imshow(firing_fields)

        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '.png', dpi=1000)
        plt.close()


def main():
    prm = PostSorting.parameters.Parameters()
    prm.set_pixel_ratio(440)
    prm.set_sampling_rate(30000)
    prm.set_local_recording_folder_path('C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of')
    firing_rate_maps = np.load('C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of/M5_2018-03-06_15-34-44_of.npy')
    spatial_firing = pd.read_pickle(prm.get_local_recording_folder_path() + '/spatial_firing.pkl')
    spatial_data = pd.read_pickle(prm.get_local_recording_folder_path() + '/position.pkl')
    # make_combined_figure(prm, spatial_firing)

    # plot_spikes_on_trajectory(spatial_data, spatial_firing, prm)
    #spatial_firing['firing_maps'] = list(firing_rate_maps)
    spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing)
    plot_hd_for_firing_fields(spatial_firing, spatial_data, prm)
if __name__ == '__main__':
    main()
