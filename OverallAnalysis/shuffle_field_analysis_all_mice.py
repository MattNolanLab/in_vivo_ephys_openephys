import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import scipy.stats

# this data frame contains results calculated by shuffle_field_analysis.py combined by load_data_frames.py
local_path_to_shuffled_field_data = '/Users/s1466507/Documents/Ephys/recordings/shuffled_field_data_all_mice.pkl'

# this is a list of fields included in the analysis with session_ids cluster ids and field ids
list_of_accepted_fields_path = '/Users/s1466507/Documents/Ephys/recordings/included_fields_detector2.csv'


def get_accepted_fields(shuffled_field_data):
    accepted_fields = pd.read_csv(list_of_accepted_fields_path)
    shuffled_field_data['field_id_unique'] = shuffled_field_data.session_id + '_' + shuffled_field_data.cluster_id.apply(str) + '_' + (shuffled_field_data.field_id + 1).apply(str)
    accepted_fields['field_id_unique'] = accepted_fields['Session ID'] + '_' + accepted_fields.Cell.apply(str) + '_' + accepted_fields.field.apply(str)

    accepted = shuffled_field_data.field_id_unique.isin(accepted_fields.field_id_unique)
    shuffled_field_data = shuffled_field_data[accepted]

    return shuffled_field_data


def find_tail_of_shuffled_distribution_of_rejects(shuffled_field_data):
    number_of_rejects = shuffled_field_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for field in number_of_rejects:
        flat_shuffled.extend(field)
    tail = max(flat_shuffled)
    percentile_95 = np.percentile(flat_shuffled, 95)
    percentile_99 = np.percentile(flat_shuffled, 99)
    return tail, percentile_95, percentile_99


def plot_histogram_of_number_of_rejected_bars(shuffled_field_data):
    number_of_rejects = shuffled_field_data.number_of_different_bins
    fig, ax = plt.subplots()
    plt.hist(number_of_rejects)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Number of rejected bars')
    ax.set_ylabel('Number of fields')
    plt.savefig('/Users/s1466507/Documents/Ephys/recordings/distribution_of_rejects.png')
    plt.close()


def plot_histogram_of_number_of_rejected_bars_shuffled(shuffled_field_data):
    number_of_rejects = shuffled_field_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for field in number_of_rejects:
        flat_shuffled.extend(field)
    fig, ax = plt.subplots()
    plt.hist(flat_shuffled, color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Number of rejected bars')
    ax.set_ylabel('Number shuffles')
    plt.savefig('/Users/s1466507/Documents/Ephys/recordings/distribution_of_rejects_shuffled.png')
    plt.close()


def make_combined_plot_of_distributions(shuffled_field_data):
    tail, percentile_95, percentile_99 = find_tail_of_shuffled_distribution_of_rejects(shuffled_field_data)

    number_of_rejects_shuffled = shuffled_field_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for field in number_of_rejects_shuffled:
        flat_shuffled.extend(field)
    fig, ax = plt.subplots()
    plt.hist(flat_shuffled, normed=True, color='black', alpha=0.5)

    number_of_rejects_real = shuffled_field_data.number_of_different_bins
    plt.hist(number_of_rejects_real, normed=True, color='navy', alpha=0.5)

    plt.axvline(x=tail, color='red', alpha=0.5, linestyle='dashed')
    plt.axvline(x=percentile_95, color='red', alpha=0.5, linestyle='dashed')
    plt.axvline(x=percentile_99, color='red', alpha=0.5, linestyle='dashed')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Number of rejected bars')
    ax.set_ylabel('Proportion')
    plt.savefig('/Users/s1466507/Documents/Ephys/recordings/distribution_of_rejects_combined_all.png')
    plt.close()


def plot_number_of_significant_p_values(field_data, type='bh'):
    if type == 'bh':
        number_of_significant_p_values = field_data.number_of_different_bins_bh
    else:
        number_of_significant_p_values = field_data.number_of_different_bins_holm

    fig, ax = plt.subplots()
    plt.hist(number_of_significant_p_values, normed='True', color='yellow', alpha=0.5)
    flat_shuffled = []
    for field in field_data.number_of_different_bins_shuffled_corrected_p:
        flat_shuffled.extend(field)
    plt.hist(flat_shuffled, normed='True', color='gray', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Number of rejected bars')
    ax.set_ylabel('Number of p values')
    plt.savefig('/Users/s1466507/Documents/Ephys/recordings/distribution_of_rejects_significant_p_ ' + type +'.png')
    plt.close()


def compare_distributions(x, y):
    stat, p = scipy.stats.mannwhitneyu(x, y)
    return p


def compare_shuffled_to_real_data_mw_test(field_data, analysis_type='bh'):
    if analysis_type == 'bh':
        flat_shuffled = []
        for field in field_data.number_of_different_bins_shuffled_corrected_p:
            flat_shuffled.extend(field)
            p_bh = compare_distributions(field_data.number_of_different_bins_bh, flat_shuffled)
            print('p value for comparing shuffled distribution to B-H corrected p values: ' + str(p_bh))
            return p_bh

    if analysis_type == 'percentile':
        flat_shuffled = []
        for field in field_data.number_of_different_bins_shuffled:
            flat_shuffled.extend(field)
            p_percentile = compare_distributions(field_data.number_of_different_bins, flat_shuffled)
            print('p value for comparing shuffled distribution to percentile thresholded p values: ' + str(p_percentile))
            return p_percentile


def main():
    shuffled_field_data = pd.read_pickle(local_path_to_shuffled_field_data)
    shuffled_field_data = get_accepted_fields(shuffled_field_data)

    plot_histogram_of_number_of_rejected_bars(shuffled_field_data)
    plot_histogram_of_number_of_rejected_bars_shuffled(shuffled_field_data)
    plot_number_of_significant_p_values(shuffled_field_data, type='bh')
    plot_number_of_significant_p_values(shuffled_field_data, type='holm')
    make_combined_plot_of_distributions(shuffled_field_data)

    compare_shuffled_to_real_data_mw_test(shuffled_field_data, analysis_type='bh')
    compare_shuffled_to_real_data_mw_test(shuffled_field_data, analysis_type='percentile')


if __name__ == '__main__':
    main()
