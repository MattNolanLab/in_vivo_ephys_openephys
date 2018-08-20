import matplotlib.pylab as plt
import plot_utility
import PostSorting.parameters

prm = PostSorting.parameters.Parameters()


def plot_spikes_on_track(spatial_firing):
    print('I am plotting spike rastas...')

    for cluster in range(len(spatial_firing)):
        spikes_on_track = plt.figure()
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        ax.plot(spatial_firing.position_cm[cluster], spatial_firing.trial_number[cluster], '|', color='black', markersize=12)
        plt.show()

        plt.ylabel('Spikes on trials', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plot_utility.style_track_plot(ax)
        x_max = max(spatial_firing.trial_number[cluster])+0.5
        plot_utility.style_vr_plot(ax, x_max)

        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/' + spatial_firing.session_id[cluster] + 'track_firing_' + str(cluster + 1) + '.png')
        plt.close()


def plot_firing_rate_maps(spike_data):
    print('I am plotting firing rate maps...')

    for cluster in range(len(spike_data)):
        avg_spikes_on_track = plt.figure()

        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(range(40), spike_data.avg_spike_per_bin[cluster], '-')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Avg spikes', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)

        plt.xlim(0,40)
        x_max = max(spike_data.avg_spike_per_bin[cluster])+0.5
        plot_utility.style_vr_plot(ax, x_max)

        plt.savefig(prm.get_local_recording_folder_path() + '/Figures/rate_map_' + str(1) + '.png')
        plt.close()
