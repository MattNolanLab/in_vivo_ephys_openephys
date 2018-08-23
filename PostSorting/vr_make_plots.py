import matplotlib.pylab as plt
import plot_utility
import PostSorting.parameters
from scipy.interpolate import spline
import numpy as np

prm = PostSorting.parameters.Parameters()


def plot_spikes_on_track(spatial_firing):
    print('I am plotting spike rastas...')

    #for cluster in range(len(spatial_firing)):
    cluster=5
    spikes_on_track = plt.figure()
    ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    ax.plot(spatial_firing.beaconed_position_cm[cluster], spatial_firing.beaconed_trial_number[cluster], '|', color='black', markersize=12)
    ax.plot(spatial_firing.nonbeaconed_position_cm[cluster], spatial_firing.nonbeaconed_trial_number[cluster], '|', color='Red', markersize=12)

    plt.ylabel('Spikes on trials', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plot_utility.style_track_plot(ax, 200)
    x_max = max(spatial_firing.trial_number[cluster])+0.5
    plot_utility.style_vr_plot(ax, x_max)

    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/track_firing_' + str(5) + '.png')
    plt.close()


def plot_firing_rate_maps(spike_data):
    print('I am plotting firing rate maps...')
    cluster_index = 5
    #for cluster in range(len(spike_data)):
    #cluster_index = spike_data.cluster_id.values[cluster] - 1
    avg_spikes_on_track = plt.figure()
    bins=range(100)
    xnew = np.linspace(0,100,200) #300 represents number of points to make between T.min and T.max
    smooth_b = spline(bins,np.array(spike_data.avg_spike_per_bin_b[cluster_index]),xnew)
    smooth_nb = spline(bins,np.array(spike_data.avg_spike_per_bin_nb[cluster_index]),xnew)

    ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(xnew, smooth_b, '-', color='Black')
    ax.plot(xnew, smooth_nb, '-', color='Red')
    ax.locator_params(axis = 'x', nbins=3)
    ax.set_xticklabels(['0', '100', '200'])
    plt.ylabel('Avg spikes', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)

    plt.xlim(0,100)
    x_max = max(spike_data.avg_spike_per_bin_b[cluster_index])+0.1
    plot_utility.style_vr_plot(ax, x_max)
    plot_utility.style_track_plot(ax, 100)

    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/rate_map_' + str(1) + '.png')
    plt.close()

