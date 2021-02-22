import array_utility
import os
import matplotlib.pylab as plt
import math
import numpy as np
import pandas as pd
import plot_utility
import PostSorting.parameters
import PostSorting.make_plots
import scipy.ndimage


# do not use this on data from more than one session
def plot_peristimulus_raster(peristimulus_spikes: pd.DataFrame, output_path: str):
    """
    :param peristimulus_spikes: Data frame with firing times of all clusters around the stimulus
    the first two columns the session and cluster ids respectively and the rest of the columns correspond to
    sampling points before and after the stimulus. 0s mean no spike and 1s mean spike at a given point
    :param output_path: fist half of the path where the plot is saved
    """
    assert len(peristimulus_spikes.groupby('session_id')['session_id'].nunique()) == 1
    save_path = output_path + '/Figures/opto_stimulation'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    cluster_ids = np.unique(peristimulus_spikes.cluster_id)
    for cluster in cluster_ids:
        cluster_rows_boolean = peristimulus_spikes.cluster_id.astype(int) == int(cluster)
        cluster_rows_annotated = peristimulus_spikes[cluster_rows_boolean]
        cluster_rows = cluster_rows_annotated.iloc[:, 2:]
        plt.cla()
        peristimulus_figure = plt.figure()
        peristimulus_figure.set_size_inches(5, 5, forward=True)
        ax = peristimulus_figure.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        sample_times = np.argwhere(np.array(cluster_rows).astype(int) == 1)[:, 1]
        trial_numbers = np.argwhere(np.array(cluster_rows).astype(int) == 1)[:, 0]
        stimulation_start = cluster_rows.shape[1] / 2   # todo remove magic number
        stimulation_end = cluster_rows.shape[1] / 2 + 90
        ax.axvspan(stimulation_start, stimulation_end, 0, cluster_rows.shape[0], alpha=0.5, color='lightblue')
        ax.vlines(x=sample_times, ymin=trial_numbers, ymax=(trial_numbers + 1), color='black', zorder=2, linewidth=3)
        plt.xlabel('Time (sampling points)', fontsize=16)
        plt.ylabel('Trial (sampling points)', fontsize=16)
        plt.ylim(0, cluster_rows.shape[0])
        plt.xlim(0, cluster_rows.shape[1])
        plt.savefig(save_path + '/' + cluster + '_peristimulus_raster.png', dpi=300)
        plt.close()

        # plt.plot((cluster_rows.astype(int)).sum().rolling(100).sum())


def get_latencies_for_cluster(spatial_firing, cluster_id):
    cluster = spatial_firing[spatial_firing.cluster_id == int(cluster_id)]
    latencies_mean = np.round(cluster.opto_latencies_mean_ms, 2)
    latencies_sd = np.round(cluster.opto_latencies_sd_ms, 2)
    if len(latencies_mean) > 0:
        return pd.to_numeric(latencies_mean).iloc[0], pd.to_numeric(latencies_sd).iloc[0]
    else:
        return pd.to_numeric(latencies_mean), pd.to_numeric(latencies_sd)


def plot_peristimulus_histogram(spatial_firing: pd.DataFrame, peristimulus_spikes: pd.DataFrame, output_path: str):
    """
    :param spatial_firing: Data frame with firing data for each cluster
    :param peristimulus_spikes: Data frame with firing times of all clusters around the stimulus
    :param output_path: fist half of the path where the plot is saved
    the first two columns the session and cluster ids respectively and the rest of the columns correspond to
    sampling points before and after the stimulus. 0s mean no spike and 1s mean spike at a given point
    """
    assert len(peristimulus_spikes.groupby('session_id')['session_id'].nunique()) == 1
    save_path = output_path + '/Figures/opto_stimulation'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    cluster_ids = np.unique(peristimulus_spikes.cluster_id)
    for cluster in cluster_ids:
        cluster_rows_boolean = peristimulus_spikes.cluster_id.astype(int) == int(cluster)
        cluster_rows_annotated = peristimulus_spikes[cluster_rows_boolean]
        cluster_rows = cluster_rows_annotated.iloc[:, 2:]
        cluster_rows = cluster_rows.astype(int).to_numpy()
        plt.cla()
        peristimulus_figure = plt.figure()
        peristimulus_figure.set_size_inches(5, 5, forward=True)
        ax = peristimulus_figure.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        number_of_spikes_per_sampling_point = np.array(np.sum(cluster_rows, axis=0))
        stimulation_start = cluster_rows.shape[1] / 2   # todo remove magic number
        stimulation_end = cluster_rows.shape[1] / 2 + 90
        latencies_mean, latencies_sd = get_latencies_for_cluster(spatial_firing, cluster)
        ax.axvspan(stimulation_start, stimulation_end, 0, np.max(number_of_spikes_per_sampling_point), alpha=0.5, color='lightblue')
        # ax.plot(number_of_spikes_per_sampling_point, color='gray', alpha=0.5)
        # convert to indices so we can make histogram
        spike_indices = np.where(cluster_rows.flatten() == 1)[0] % len(number_of_spikes_per_sampling_point)
        plt.hist(spike_indices, color='grey', alpha=0.5, bins=50)
        plt.xlabel('Time (sampling points)', fontsize=16)
        plt.ylabel('Number of spikes', fontsize=16)
        #plt.ylim(0, np.max(number_of_spikes_per_sampling_point) + 2)
        plt.xlim(0, len(number_of_spikes_per_sampling_point))
        plt.title('Mean latency: ' + str(latencies_mean) + ' ms, sd = ' + str(latencies_sd))
        plt.savefig(save_path + '/' + cluster + '_peristimulus_histogram.png', dpi=300)
        plt.close()


def plot_waveforms_opto(spike_data, prm, snippets_column_name='random_snippets_opto'):
    if snippets_column_name in spike_data:
        print('I will plot the waveform shapes for each cluster for opto_tagging data.')
        save_path = prm.get_output_path() + '/Figures/opto_stimulation'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
            cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

            max_channel = cluster_df['primary_channel'].iloc[0]
            fig = plt.figure(figsize=(5, 5))
            grid = plt.GridSpec(2, 2, wspace=0.5, hspace=0.5)
            for channel in range(4):
                PostSorting.make_plots.plot_spikes_for_channel_centered(grid, spike_data, cluster_id, channel, snippets_column_name)

            plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + snippets_column_name + '.png', dpi=300, bbox_inches='tight', pad_inches=0)
            # plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster + 1) + '_waveforms_opto.pdf', bbox_inches='tight', pad_inches=0)
            plt.close()


def make_optogenetics_plots(spatial_firing, prm):
    peristimulus_spikes_path = prm.get_output_path() + '/DataFrames/peristimulus_spikes.pkl'
    if os.path.exists(peristimulus_spikes_path):
        peristimulus_spikes = pd.read_pickle(peristimulus_spikes_path)
        output_path = prm.get_output_path()
        plot_peristimulus_raster(peristimulus_spikes, output_path)
        plot_peristimulus_histogram(spatial_firing, peristimulus_spikes, output_path)
        plot_waveforms_opto(spatial_firing, prm, snippets_column_name='random_snippets_opto')
        plot_waveforms_opto(spatial_firing, prm, snippets_column_name='first_spike_snippets_opto')



def main():
    prm = PostSorting.parameters.Parameters()
    path = 'C:/Users/s1466507/Documents/Ephys/recordings/M0_2017-12-14_15-00-13_of/MountainSort/DataFrames/peristimulus_spikes.pkl'
    peristimulus_spikes = pd.read_pickle(path)
    prm.set_output_path('C:/Users/s1466507/Documents/Ephys/recordings/M0_2017-12-14_15-00-13_of/MountainSort/')
    output_path = prm.get_output_path()
    plot_peristimulus_histogram(peristimulus_spikes, output_path)
    plot_peristimulus_raster(peristimulus_spikes, output_path)


if __name__ == '__main__':
    main()
