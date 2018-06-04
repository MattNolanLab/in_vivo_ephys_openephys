import matplotlib.pylab as plt
import numpy as np
import plot_utility


def plot_firing_rate_hist(spike_data_frame, save_output_path):
    fr_fig = plt.figure()
    ax = fr_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    fr_fig, ax = plot_utility.style_plot(ax)
    ax.hist(spike_data_frame.avgFR, bins=20, color='navy')
    plt.xlabel('Average firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(save_output_path + 'avg_firing_rate_histogram.png')


def plot_max_fr_spatial(spike_data_frame, save_output_path):
    max_fr_spatial = plt.figure()
    ax = max_fr_spatial.add_subplot(111)
    fr_fig, ax = plot_utility.style_plot(ax)
    bins = np.linspace(0, 200, 20)
    ax.hist(spike_data_frame.maxFRspatial, bins, color='navy')
    plt.xlabel('Maximum firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(save_output_path + 'max_firing_rate_histogram_spatial.png')


def plot_max_fr_spatial_excitatory(spike_data_frame, save_output_path):
    max_fr_spatial = plt.figure()
    ax = max_fr_spatial.add_subplot(111)
    fr_fig, ax = plot_utility.style_plot(ax)
    bins = np.linspace(0, 20, 20)
    ax.hist(spike_data_frame.maxFRspatial, bins, color='navy')
    plt.xlabel('Maximum firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(save_output_path + 'excitatory_max_firing_rate_histogram_spatial.png')


def plot_max_fr_head_dir(spike_data_frame, save_output_path):
    max_fr_spatial = plt.figure()
    ax = max_fr_spatial.add_subplot(111)
    fr_fig, ax = plot_utility.style_plot(ax)
    max_fr = max(spike_data_frame.HD_maxFR.values)
    bins = np.linspace(0, 200, 20)
    ax.hist(spike_data_frame.HD_maxFR, bins, color='navy')
    plt.xlabel('Maximum firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(save_output_path + 'max_firing_rate_histogram_head_dir.png')


def plot_max_fr_head_dir_excitatory(spike_data_frame, save_output_path):
    max_fr_spatial = plt.figure()
    ax = max_fr_spatial.add_subplot(111)
    fr_fig, ax = plot_utility.style_plot(ax)
    max_fr = max(spike_data_frame.HD_maxFR.values)
    bins = np.linspace(0, 20, 20)
    ax.hist(spike_data_frame.HD_maxFR, bins, color='navy')
    plt.xlabel('Maximum firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(save_output_path + 'excitatory_max_firing_rate_histogram_head_dir.png')


def plot_grid_score_hist(spike_data_frame, save_output_path):
    grid_sc_fig = plt.figure()
    ax = grid_sc_fig.add_subplot(1, 1, 1)
    fr_fig, ax = plot_utility.style_plot(ax)
    has_grid_score = spike_data_frame['gridscore'].notnull()
    ax.hist(spike_data_frame[has_grid_score].gridscore, color='navy')
    plt.xlabel('Grid score')
    plt.ylabel('Number of cells')
    plt.savefig(save_output_path + 'grid_score_histogram.png')


def plot_grid_score_vs_firing_rate(spike_data_frame, save_output_path):
    grid_sc_vs_fr_fig = plt.figure()
    ax = grid_sc_vs_fr_fig.add_subplot(1, 1, 1)
    fr_fig, ax = plot_utility.style_plot(ax)
    has_grid_score = spike_data_frame['gridscore'].notnull()
    x = spike_data_frame[has_grid_score].avgFR.values
    y = spike_data_frame[has_grid_score].gridscore.values
    ax.plot(x, y, 'o', color='navy')
    plt.xlabel('Average firing rate')
    plt.ylabel('Grid score')
    plt.savefig(save_output_path + 'grid_score_vs_avgFR.png')


def plot_hd_score_vs_firing_rate(spike_data_frame, save_output_path):
    grid_sc_vs_fr_fig = plt.figure()
    ax = grid_sc_vs_fr_fig.add_subplot(1, 1, 1)
    fr_fig, ax = plot_utility.style_plot(ax)
    x = spike_data_frame.avgFR.values
    y = spike_data_frame.HD_maxFR.values
    ax.plot(x, y, 'o', color='navy')
    plt.xlabel('Average firing rate')
    plt.ylabel('Head-direction score')
    plt.savefig(save_output_path + 'hd_score_vs_avgFR.png')


def plot_grid_score_vs_hd_score(spike_data_frame, save_output_path):
    grid_sc_vs_fr_fig = plt.figure()
    ax = grid_sc_vs_fr_fig.add_subplot(1, 1, 1)
    fr_fig, ax = plot_utility.style_plot(ax)
    has_grid_score = spike_data_frame['gridscore'].notnull()
    x = spike_data_frame[has_grid_score].HD_maxFR.values
    y = spike_data_frame[has_grid_score].gridscore.values
    ax.plot(x, y, 'o', color='navy')
    plt.xlabel('Head-direction score')
    plt.ylabel('Grid score')
    plt.savefig(save_output_path + 'grid_score_vs_hd_score.png')


def plot_spatial_coherence_vs_firing_rate(spike_data_frame, save_output_path):
    grid_sc_vs_fr_fig = plt.figure()
    ax = grid_sc_vs_fr_fig.add_subplot(1, 1, 1)
    fr_fig, ax = plot_utility.style_plot(ax)
    x = spike_data_frame.avgFR.values
    y = spike_data_frame.spatialcoherence.values
    ax.plot(x, y, 'o', color='navy')
    plt.xlabel('Average firing rate')
    plt.ylabel('Spatial coherence')
    plt.savefig(save_output_path + 'spatial_coherence_vs_avgFR.png')


def plot_all(spike_data_frame, save_output_path):
    plot_firing_rate_hist(spike_data_frame, save_output_path)
    plot_max_fr_spatial(spike_data_frame, save_output_path)
    plot_max_fr_spatial_excitatory(spike_data_frame, save_output_path)
    plot_max_fr_head_dir(spike_data_frame, save_output_path)
    plot_max_fr_head_dir_excitatory(spike_data_frame, save_output_path)
    plot_grid_score_hist(spike_data_frame, save_output_path)
    plot_grid_score_vs_firing_rate(spike_data_frame, save_output_path)
    plot_hd_score_vs_firing_rate(spike_data_frame, save_output_path)
    plot_grid_score_vs_hd_score(spike_data_frame, save_output_path)
    plot_spatial_coherence_vs_firing_rate(spike_data_frame, save_output_path)

