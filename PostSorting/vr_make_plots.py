import matplotlib.pylab as plt
import plot_utility
import PostSorting.parameters
import numpy as np

prm = PostSorting.parameters.Parameters()


def plot_stops_on_track(spatial_data):
    print('I am plotting stop rasta...')
    spikes_on_track = plt.figure(figsize=(6,8))
    ax = spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)

    ax.plot(spatial_data.stop_location_cm, spatial_data.stop_trial_number, 'o', color='0.3', markersize=4, alpha = 0.2)
    ax.plot(spatial_data.first_series_location_cm, spatial_data.first_series_trial_number, 'o', color='Black', markersize=4)
    #ax.plot(spatial_data.rewarded_stop_locations, spatial_data.rewarded_trials, '>', color='Red', markersize=4)
    plt.ylabel('Spikes on trials', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(spatial_data.trial_number)+0.5
    plot_utility.style_vr_plot(ax, x_max)

    ax = spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)

    ax.plot(spatial_data.position_bins,spatial_data.average_stops, '-', color='Black')

    plt.ylabel('Spikes on trials', fontsize=14, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(spatial_data.average_stops)+0.5
    plot_utility.style_vr_plot(ax, x_max)

    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/stops_on_track_' + '.png')
    plt.close()


def plot_spikes_on_track(spike_data,spatial_data):
    print('I am plotting spike rastas...')
    #cluster_index = 5
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        spikes_on_track = plt.figure()
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        cluster_firing_indices = spike_data.firing_times[cluster_index]
        ax.plot(spatial_data.x_position_cm[cluster_firing_indices], spatial_data.trial_number[cluster_firing_indices], '|', color='Black', markersize=8)
        ax.plot(spike_data.loc[cluster_index].nonbeaconed_position_cm, spike_data.loc[cluster_index].nonbeaconed_trial_number, '|', color='Red', markersize=8)
        ax.plot(spike_data.loc[cluster_index].probe_position_cm, spike_data.loc[cluster_index].probe_trial_number, '|', color='Blue', markersize=8)

        plt.ylabel('Spikes on trials', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plot_utility.style_track_plot(ax, 200)
        x_max = max(spatial_data.trial_number[cluster_firing_indices])+0.5
        plot_utility.style_vr_plot(ax, x_max)

        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/track_firing_Cluster_' + str(cluster_index +1) + '.png')
        plt.close()

def plot_firing_rate_maps(spike_data):
    print('I am plotting firing rate maps...')
    #cluster_index = 5
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        avg_spikes_on_track = plt.figure()

        bins=range(200)

        unsmooth_b = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_b'])
        unsmooth_nb = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_nb'])
        unsmooth_p = np.array(spike_data.at[cluster_index, 'avg_spike_per_bin_p'])

        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(bins, unsmooth_b, '-', color='Black')
        try:
            ax.plot(bins, unsmooth_nb, '-', color='Red')
            ax.plot(bins, unsmooth_p, '-', color='Blue')
        except ValueError:
            continue
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        x_max = max(spike_data.avg_spike_per_bin_b[cluster_index])+0.1
        plot_utility.style_vr_plot(ax, x_max)
        plot_utility.style_track_plot(ax, 200)

        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/rate_map_Cluster_' + str(cluster_index +1) + '.png')
        plt.close()

