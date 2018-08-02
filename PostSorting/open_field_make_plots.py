import matplotlib.pylab as plt
import os
import matplotlib.cm as cm


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
        hd_plot = ax.scatter(x_positions, y_positions, s=20, c=hd, vmin=-180, vmax=180, marker='o', cmap='jet')
        plt.colorbar(hd_plot)
        ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1,
                alpha=0.7)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_map_' + str(cluster + 1) + '.png')
        hd_plot = ax.scatter(x_positions, y_positions, s=20, c=hd, vmin=-180, vmax=180, marker='o')
        ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1,
                alpha=0.7)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_map2_' + str(cluster + 1) + '.png')
        plt.close()