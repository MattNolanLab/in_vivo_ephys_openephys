import os
import matplotlib.pylab as plt
import math
import numpy as np
import plot_utility


def plot_spike_histogram(spatial_firing, prm):
    sampling_rate = prm.get_sampling_rate()
    print('I will plot spikes vs time for the whole session excluding opto tagging.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        firings_cluster = spatial_firing.firing_times[cluster]
        spike_hist = plt.figure()
        spike_hist.set_size_inches(5, 5, forward=True)
        ax = spike_hist.add_subplot(1, 1, 1)
        spike_hist, ax = plot_utility.style_plot(ax)
        number_of_bins = int((firings_cluster[-1] - firings_cluster[0]) / (5*sampling_rate))
        if number_of_bins > 0:
            hist, bins = np.histogram(firings_cluster, bins=number_of_bins)
            width = bins[1] - bins[0]
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width, color='black')
        plt.title('total spikes = ' + str(spatial_firing.number_of_spikes[cluster]) + ', mean fr = ' + str(round(spatial_firing.mean_firing_rate[cluster], 0)) + ' Hz', y=1.08)
        plt.xlabel('time (sampling points)')
        plt.ylabel('number of spikes')
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_histogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_firing_rate_vs_speed(spatial_firing, spatial_data,  prm):
    sampling_rate = 30
    print('I will plot spikes vs speed for the whole session excluding opto tagging.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    number_of_bins = math.ceil(max(spatial_data.speed)) - math.floor(min(spatial_data.speed))
    session_hist, bins_s = np.histogram(spatial_data.speed, bins=number_of_bins, range=(math.floor(min(spatial_data.speed)), math.ceil(max(spatial_data.speed))))
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        speed_cluster = spatial_firing.speed[cluster]
        speed_cluster = sorted(speed_cluster)
        spike_hist = plt.figure()
        spike_hist.set_size_inches(5, 5, forward=True)
        ax = spike_hist.add_subplot(1, 1, 1)
        speed_hist, ax = plot_utility.style_plot(ax)
        if number_of_bins > 0:
            hist, bins = np.histogram(speed_cluster[1:], bins=number_of_bins, range=(math.floor(min(spatial_data.speed)), math.ceil(max(spatial_data.speed))))
            width = bins[1] - bins[0]
            center = (bins[:-1] + bins[1:]) / 2
            center = center[[np.where(session_hist > sum(session_hist)*0.005)]]
            rate = hist/session_hist
            rate = rate[[np.where(session_hist > sum(session_hist)*0.005)]]
            plt.bar(center[0], rate[0]*sampling_rate, align='center', width=width, color='black')
        plt.xlabel('speed [cm/s]')
        plt.ylabel('firing rate [Hz]')
        plt.xlim(0, 30)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_speed_histogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


def calculate_autocorrelogram_hist(spikes, bin_size, window):
    number_of_bins = int(math.ceil(spikes[-1]*1000))
    train = np.zeros(number_of_bins)
    bins = np.zeros(len(spikes))

    for spike in range(len(spikes)):
        bin = math.floor(spikes[spike]*1000)
        train[bin] = train[bin] + 1
        bins[spike] = bin

    counts = np.zeros(window+1)
    counted = 0
    for b in range(len(bins)):
        bin = bins[b]
        if (bin > (window/2) + 1) and (bin < len(train) - window/2):
            counts = counts + train[bin - window/2 :bin + window/2+1]
            counted = counted + sum(train[bin-window/2 - 1:bin + window/2]) - train[bin]

    counts[window/2] = 0
    if max(counts) == 0 and counted == 0:
        counted = 1

    corr = counts / counted
    time = np.arange(-window/2, window/2 + 1, bin_size)
    return corr, time


def plot_autocorrelograms(spike_data, prm):
    print('I will plot autocorrelograms for each cluster.')
    save_path = prm.get_local_recording_folder_path() + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster = spike_data.cluster_id.values[cluster] - 1
        firing_times_cluster = spike_data.firing_times[cluster]
        #lags = plt.acorr(firing_times_cluster, maxlags=firing_times_cluster.size-1)
        corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 20)

        plt.bar(time, corr, align='center', width=1, color='black')
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster + 1) + '_autocorrelogram_10ms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.figure()
        corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster)/prm.get_sampling_rate(), 1, 500)
        plt.bar(time, corr, align='center', width=1, color='black')
        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster + 1) + '_autocorrelogram_250ms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
