import os
import matplotlib.pylab as plt
import numpy as np
import plot_utility


def plot_spike_histogram(spatial_firing, prm):
    print('I will make scatter plots of spikes on the trajectory of the animal.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        firings_cluster = spatial_firing.firing_times[cluster]
        spike_hist = plt.figure()
        spike_hist.set_size_inches(5, 2.5, forward=True)
        ax = spike_hist.add_subplot(1, 1, 1)
        spike_hist, ax = plot_utility.style_plot(ax)
        number_of_bins = int((firings_cluster[-1] - firings_cluster[0]) / (5*30000))
        hist, bin_edges = np.histogram(firings_cluster, bins=number_of_bins)
        ax.hist(hist, len(hist), color='black')
        plt.title('total spikes = ' + str(spatial_firing.number_of_spikes[cluster]) + ', mean fr = ' + str(round(spatial_firing.mean_firing_rate[cluster] + ' Hz', 0)), y=1.08)
        plt.xlabel('sampling points (30000 per second)')
        plt.ylabel('Number of spikes')
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_hitogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
