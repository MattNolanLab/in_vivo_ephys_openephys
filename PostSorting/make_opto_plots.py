import array_utility
import os
import matplotlib.pylab as plt
import math
import numpy as np
import pandas as pd
import plot_utility
import PostSorting.parameters
import scipy.ndimage


# do not use this on data from more than one session
def plot_peristimulus_raster(peristimulus_spikes, output_path):
    assert len(peristimulus_spikes.groupby('session_id')['session_id'].nunique()) == 1
    save_path = output_path + '/Figures/opto_stimulation'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    cluster_ids = np.unique(peristimulus_spikes.cluster_id)
    for cluster in cluster_ids:
        cluster_rows_boolean = peristimulus_spikes.cluster_id.astype(int) == int(cluster)
        cluster_rows_annotated = peristimulus_spikes[cluster_rows_boolean]
        cluster_rows = cluster_rows_annotated.iloc[:, 2:]
        print(cluster_rows.head())
        plt.cla()
        peristimulus_figure = plt.figure()
        peristimulus_figure.set_size_inches(5, 5, forward=True)
        ax = peristimulus_figure.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        sample_times = np.argwhere(np.array(cluster_rows).astype(int) == 1)[:, 1]
        trial_numbers = np.argwhere(np.array(cluster_rows).astype(int) == 1)[:, 0]
        stimulation_start = cluster_rows.shape[1] / 2 - 45  # todo remove magic number
        stimulation_end = cluster_rows.shape[1] / 2 + 45
        ax.axvspan(stimulation_start, stimulation_end, 0, cluster_rows.shape[0], alpha=0.5, color='lightblue')
        ax.vlines(x=sample_times, ymin=trial_numbers, ymax=(trial_numbers + 1), color='black', zorder=2, linewidth=3)
        plt.xlabel('Time (sampling points)')
        plt.ylabel('Trial (sampling points)')
        plt.ylim(0, cluster_rows.shape[0])
        plt.savefig(save_path + '/' + cluster + '_peristimulus_raster.png', dpi=300)
        plt.close()

        # plt.plot((cluster_rows.astype(int)).sum().rolling(100).sum())


def plot_peristimulus_histogram(peristimulus_spikes, output_path):
    assert len(peristimulus_spikes.groupby('session_id')['session_id'].nunique()) == 1
    save_path = output_path + '/Figures/opto_stimulation'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    cluster_ids = np.unique(peristimulus_spikes.cluster_id)
    for cluster in cluster_ids:
        cluster_rows_boolean = peristimulus_spikes.cluster_id.astype(int) == int(cluster)
        cluster_rows_annotated = peristimulus_spikes[cluster_rows_boolean]
        cluster_rows = cluster_rows_annotated.iloc[:, 2:]
        print(cluster_rows.head())
        plt.cla()
        peristimulus_figure = plt.figure()
        peristimulus_figure.set_size_inches(5, 5, forward=True)
        ax = peristimulus_figure.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        number_of_spikes_per_samplng_point = np.sum(cluster_rows, axis=0)
        stimulation_start = cluster_rows.shape[1] / 2 - 45  # todo remove magic number
        stimulation_end = cluster_rows.shape[1] / 2 + 45
        ax.axvspan(stimulation_start, stimulation_end, 0, cluster_rows.shape[0], alpha=0.5, color='lightblue')
        plt.plot(number_of_spikes_per_samplng_point, color='gray', alpha=0.5)
        plt.xlabel('Time (sampling points)', fontsize=16)
        plt.ylabel('Number of spikes', fontsize=16)
        plt.ylim(0, np.max(number_of_spikes_per_samplng_point) + 10)
        plt.savefig(save_path + '/' + cluster + '_peristimulus_histogram.png', dpi=300)
        plt.close()


def make_optogenetics_plots(prm):
    peristimulus_spikes_path = prm.get_output_path() + '/DataFrames/peristimulus_spikes.pkl'
    if os.path.exists(peristimulus_spikes_path):
        peristimulus_spikes = pd.read_pickle(peristimulus_spikes_path)
        output_path = prm.get_output_path()
        plot_peristimulus_raster(peristimulus_spikes, output_path)


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
