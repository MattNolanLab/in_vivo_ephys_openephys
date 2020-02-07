'''
I will quantify how similar the trajectory hd distribution is to a uniform distribution (1 sample watson test)
 and then correlate the results of this to the number of significant bins from the distributive shuffled analysis
'''


import matplotlib.pylab as plt
import numpy as np
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_field_analysis_all_animals
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import scipy.stats
# utils = importr('utils')
# utils.install_packages('circular')


analysis_path = OverallAnalysis.folder_path_settings.get_local_path() + '/tuning_bias_vs_speed/'
local_path_to_shuffled_field_data_mice = analysis_path + 'shuffled_field_data_all_mice.pkl'
local_path_to_shuffled_field_data_rats = analysis_path + 'shuffled_field_data_all_rats.pkl'


# run 2 sample watson test and put it in df
def run_one_sample_watson_test(hd_session):
    circular = importr("circular")
    watson_test = circular.watson_test
    hd_session = ro.FloatVector(hd_session)
    stat = watson_test(hd_session)
    return stat[0][0]  # this is the part of the return r object that is the stat


def compare_trajectory_hd_to_uniform_dist(fields):
    hd = fields.hd_in_field_session
    stats_values = []
    for field in hd:
        stat = run_one_sample_watson_test(field)
        stats_values.append(stat)
    fields['watson_stat'] = stats_values
    return fields


def plot_results(grid_fields, animal):
    number_of_significantly_directional_bins = grid_fields.number_of_different_bins_bh
    speed_score = grid_fields.speed_score
    plt.figure()
    plt.scatter(speed_score, number_of_significantly_directional_bins)
    # plt.axvline(x=0.1, color='red')
    plt.ylabel('Number of significant bins', fontsize=16)
    plt.xlabel('Speed score', fontsize=16)
    plt.xticks([-0.2, -0.1, 0, 0.1, 0.2], fontsize=12)
    plt.yticks([0, 5, 10, 15], fontsize=12)
    plt.savefig(analysis_path + 'number_of_significantly_directional_bins_vs_speed_score' + animal + '.png')
    plt.cla()

    plt.figure()
    plt.scatter(speed_score, grid_fields.directional_percentile)
    # plt.axvline(x=0.1, color='red')
    plt.ylabel('Directional percentile', fontsize=16)
    plt.xlabel('Speed score', fontsize=16)
    plt.savefig(analysis_path + 'directional_percentile_vs_speed_score' + animal + '.png')
    plt.cla()


def add_percentiles(fields):
    percentiles_correction = []
    for index, field in fields.iterrows():
        percentile = scipy.stats.percentileofscore(field.number_of_different_bins_shuffled_corrected_p, field.number_of_different_bins_bh)
        percentiles_correction.append(percentile)
    fields['directional_percentile'] = percentiles_correction
    return fields


def add_speed_score_to_df(fields, animal):
    fields['cell_id'] = fields.session_id + fields.cluster_id.astype(str)
    if animal == 'mouse':
        df = pd.read_pickle(analysis_path + 'all_mice_df.pkl')
    else:
        df = pd.read_pickle(analysis_path + 'all_rats_df.pkl')
    df['cell_id'] = df.session_id + df.cluster_id.astype(str)
    df = df[['cell_id', 'speed_score']]
    fields = pd.merge(fields, df, on='cell_id')
    return fields


def compare_speed_modulated_and_non_modulated(grid_fields, animal, speed_threshold=0.1):
    print('Speed threshold is: ' + str(speed_threshold))
    speed_mod = grid_fields[grid_fields.speed_score > speed_threshold]
    not_speed_mod = grid_fields[grid_fields.speed_score <= speed_threshold]
    dir_bins_speed = speed_mod.number_of_different_bins_bh
    dir_bins_not_speed = not_speed_mod.number_of_different_bins_bh
    D, p = scipy.stats.mannwhitneyu(dir_bins_speed, dir_bins_not_speed)
    print('_______________________')
    print(animal)
    print('Compare number of significant bins for speed dep and non speed dep')
    print('MW test on number of significant bins (U, p): ')
    print(D)
    print(p)

    percentile_speed_mod = speed_mod.directional_percentile
    percentile_not_speed_mod = not_speed_mod.directional_percentile

    print('Number of directional fields that are speed modulated:')
    print(len(speed_mod[speed_mod.directional_percentile > 95]))
    print('Number of significant bins (mean and SD):')
    print(speed_mod[speed_mod.directional_percentile > 95].number_of_different_bins_bh.mean())
    print(speed_mod[speed_mod.directional_percentile > 95].number_of_different_bins_bh.std())
    print('Number of non directional fields that are speed modulated:')
    print(len(speed_mod[speed_mod.directional_percentile <= 95]))


    print('Number of directional fields that are not speed modulated:')
    print(len(not_speed_mod[not_speed_mod.directional_percentile > 95]))
    print('Number of significant bins (mean and SD):')
    print(not_speed_mod[not_speed_mod.directional_percentile > 95].number_of_different_bins_bh.mean())
    print(not_speed_mod[not_speed_mod.directional_percentile > 95].number_of_different_bins_bh.std())
    print('Number of non directional fields that are not speed modulated:')
    print(len(not_speed_mod[not_speed_mod.directional_percentile <= 95]))

    D, p = scipy.stats.mannwhitneyu(percentile_speed_mod, percentile_not_speed_mod)
    print('MW test on directional percentiles (U, p): ')
    print(D)
    print(p)


def process_data(animal):
    if animal == 'mouse':
        local_path_to_field_data = local_path_to_shuffled_field_data_mice
        accepted_fields = pd.read_excel(analysis_path + 'list_of_accepted_fields.xlsx')
        shuffled_field_data = pd.read_pickle(local_path_to_field_data)
        shuffled_field_data = OverallAnalysis.shuffle_field_analysis_all_animals.tag_accepted_fields_mouse(shuffled_field_data, accepted_fields)

    else:
        local_path_to_field_data = local_path_to_shuffled_field_data_rats
        accepted_fields = pd.read_excel(analysis_path + 'included_fields_detector2_sargolini.xlsx')
        shuffled_field_data = pd.read_pickle(local_path_to_field_data)
        shuffled_field_data = OverallAnalysis.shuffle_field_analysis_all_animals.tag_accepted_fields_rat(shuffled_field_data, accepted_fields)

    grid = shuffled_field_data.grid_score >= 0.4
    hd = shuffled_field_data.hd_score >= 0.5
    grid_cells = np.logical_and(grid, np.logical_not(hd))
    accepted_field = shuffled_field_data.accepted_field == True
    grid_fields = shuffled_field_data[grid_cells & accepted_field]
    # grid_fields = compare_trajectory_hd_to_uniform_dist(grid_fields)
    grid_fields = add_speed_score_to_df(grid_fields, animal)
    grid_fields = add_percentiles(grid_fields)

    plot_results(grid_fields, animal)
    compare_speed_modulated_and_non_modulated(grid_fields, animal)
    compare_speed_modulated_and_non_modulated(grid_fields, animal, speed_threshold=0.05)


def main():
    process_data('mouse')
    process_data('rat')


if __name__ == '__main__':
    main()

