import matplotlib.pylab as plt


def plot_position(position_data):
    plt.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=5)
    plt.show()


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

    plt.savefig('C:/Users/s1466507/Documents/Ephys/overall_figures/' + 'spatial_firing.png')


def plot_coverage(firing_maps, prm):
    coverage = plt.figure()
    ax = coverage.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.imshow(firing_maps, cmap='jet', interpolation='nearest')
    plt.savefig('C:/Users/s1466507/Documents/Ephys/overall_figures/' + 'heatmap.png')
