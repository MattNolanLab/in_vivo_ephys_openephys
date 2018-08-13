import os
import matplotlib.pylab as plt
import numpy as np
import plot_utility


def plot_spike_histogram(spatial_firing, prm):
    print('I will plot spikes vs time for the whole session excluding opto tagging.')
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
        if number_of_bins > 0:
            hist, bins = np.histogram(firings_cluster, bins=number_of_bins)
            width = bins[1] - bins[0]
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width)
        plt.title('total spikes = ' + str(spatial_firing.number_of_spikes[cluster]) + ', mean fr = ' + str(round(spatial_firing.mean_firing_rate[cluster], 0)) + ' Hz', y=1.08)
        plt.xlabel('time (sampling points)')
        plt.ylabel('number of spikes')
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_hitogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_firing_rate_vs_speed(spatial_firing, prm):
    print('I will plot spikes vs speed for the whole session excluding opto tagging.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        speed_cluster = spatial_firing.speed[cluster]
        spike_hist = plt.figure()
        spike_hist.set_size_inches(5, 2.5, forward=True)
        ax = spike_hist.add_subplot(1, 1, 1)
        speed_hist, ax = plot_utility.style_plot(ax)
        number_of_bins = int((speed_cluster[-1] - speed_cluster[0]))
        if number_of_bins > 0:
            hist, bins = np.histogram(speed_cluster, bins=number_of_bins)
            width = bins[1] - bins[0]
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width)
        plt.xlabel('speed [cm/s]')
        plt.ylabel('firing rate [Hz]')
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_hitogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
