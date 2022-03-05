import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd


def plot_snippet_method(mean_snippet, snippet_height, half_height, intercept_line, width):
    plt.plot(mean_snippet)
    plt.plot(snippet_height, 'o', color='r', markersize=5)
    plt.plot(half_height, 'o', color='b', markersize=5)
    plt.plot(intercept_line, '-', color='r', markersize=5)
    plt.title('width= ' + str(np.round(width)))
    plt.show()


def find_intercept(mean_snippet, intercept_line):
    idx = np.argwhere(np.diff(np.sign(mean_snippet - intercept_line))).flatten()
    return idx


def extract_mean_spike_width_for_channel(mean_snippet):
    mean_snippet = mean_snippet * -1
    snippet_height = np.max(mean_snippet) - np.min(mean_snippet)
    half_height = snippet_height/2
    intercept_line = np.repeat(half_height/2, mean_snippet.shape[0])
    intercept = find_intercept(mean_snippet, intercept_line)
    try:
        width = intercept[1]-intercept[0]
    except IndexError:
        width = 0
    plot_snippet_method(mean_snippet, snippet_height, half_height, intercept_line, width)
    return width


def remove_outlier_waveforms(all_waveforms):
    # remove snippets that have data points > 3 standard dev away from mean
    mean = all_waveforms.mean(axis=1)
    sd = all_waveforms.std(axis=1)
    distance_from_mean = all_waveforms.T - mean
    max_deviations = 3
    outliers = np.sum(distance_from_mean > max_deviations * sd, axis=1) > 0
    return all_waveforms[:, ~outliers]


def add_spike_half_width_to_df(spatial_firing):
    spike_width = []
    for index, cell in spatial_firing.iterrows():
        primary_channel = cell.primary_channel - 1
        all_waveforms_with_noise = cell.random_snippets[primary_channel]
        all_waveforms = remove_outlier_waveforms(all_waveforms_with_noise)
        mean_waveform = all_waveforms.mean(axis=1)
        width = extract_mean_spike_width_for_channel(mean_waveform)
        spike_width.append(width)

    spatial_firing['spike_width'] = spike_width
    return spatial_firing


def analyse_waveform_shapes(recording_folder_path):
    print('Calculate spike half width.')
    spatial_firing_path = recording_folder_path + 'MountainSort/DataFrames/spatial_firing.pkl'
    if os.path.exists(spatial_firing_path):
        spatial_firing = pd.read_pickle(spatial_firing_path)
        # spatial_firing = add_filtered_big_snippets_to_data(recording_folder_path, spatial_firing)
        spatial_firing = add_spike_half_width_to_df(spatial_firing)
        spatial_firing.to_pickle(recording_folder_path + 'MountainSort/DataFrames/spatial_firing.pkl')

    else:
        print('There is no spatial firing data for this recording: ' + recording_folder_path)
        return False


def main():
    # there are 2 grid cells in this recording and one of them (#7) looks theta modulated
    # recording_folder_path = '/mnt/datastore/Klara/Open_field_opto_tagging_p038/M13_2018-05-14_09-37-33_of/'
    # recording_folder_path = '/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo/M10_2021-12-10_08-37-27_of/'
    recording_folder_path = '/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo_extras/PSAM/M10_2021-11-26_16-07-10_of/'
    analyse_waveform_shapes(recording_folder_path)


if __name__ == '__main__':
    main()