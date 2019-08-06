import array_utility
import os
import matplotlib.pylab as plt
import math
import numpy as np
import pandas as pd
import plot_utility
import scipy.ndimage


# do not use this on data from more than one session
def plot_peristimulus_raster(peristimulus_spikes):
    assert len(peristimulus_spikes.groupby('session_id')['session_id'].nunique()) == 1
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
        sample_times = np.argwhere(cluster_rows.to_numpy().astype(int) == 1)[:, 1]
        trial_numbers = np.argwhere(cluster_rows.to_numpy().astype(int) == 1)[:, 0]

        plt.vlines(x=sample_times, ymin=trial_numbers, ymax=(trial_numbers + 1), color='black', zorder=2)


def main():
    path = 'C:/Users/s1466507/Documents/Ephys/recordings/M0_2017-12-14_15-00-13_of/MountainSort/DataFrames/peristimulus_spikes.pkl'
    peristimulus_spikes = pd.read_pickle(path)
    plot_peristimulus_raster(peristimulus_spikes)


if __name__ == '__main__':
    main()
