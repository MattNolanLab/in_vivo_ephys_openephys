import pandas as pd
import numpy as np
import OverallAnalysis.false_positives
import OverallAnalysis.analyze_field_correlations


def load_data_frame(path):
    # this is the output of load_df.py which read all dfs from a folder and saved selected columns into a combined df
    df = pd.read_pickle(path)
    return df


def add_combined_id_to_df(df_all_mice):
    animal_ids = [session_id.split('_')[0] for session_id in df_all_mice.session_id.values]
    dates = [session_id.split('_')[1] for session_id in df_all_mice.session_id.values]
    tetrode = df_all_mice.tetrode.values
    cluster = df_all_mice.cluster_id.values

    combined_ids = []
    for cell in range(len(df_all_mice)):
        id = animal_ids[cell] +  '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    df_all_mice['false_positive_id'] = combined_ids
    return df_all_mice


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    local_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df_2.pkl'
    path_to_data = 'C:/Users/s1466507/Documents/Ephys/recordings/'
    save_output_path = 'C:/Users/s1466507/Documents/Ephys/overall_figures/'
    false_positives_path = path_to_data + 'false_positives_all.txt'
    df_all_mice = load_data_frame(local_path)
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(false_positives_path)
    df_all_mice = add_combined_id_to_df(df_all_mice)
    df_all_mice['false_positive'] = df_all_mice['false_positive_id'].isin(list_of_false_positives)

    good_cluster = df_all_mice.false_positive == False

    excitatory_neurons = df_all_mice.mean_firing_rate <= 10
    inhibitory_neurons = df_all_mice.mean_firing_rate > 10
    significant_corr = df_all_mice.hd_correlation_first_vs_second_half_p < 0.001
    print('mean pearson r of correlation between first and second half')
    print(df_all_mice.hd_correlation_first_vs_second_half[significant_corr & good_cluster].mean())

    watson_result_exists = df_all_mice.watson_test_hd.notnull()
    is_hd_cell = df_all_mice.hd_score >= 0.5
    print('Number of cells included in two sample watson test for head-direction from the whole session: ' + str(watson_result_exists.sum()))
    print('excitatory: ' + str(len(df_all_mice[excitatory_neurons & watson_result_exists])))
    print('inhibitory: ' + str(len(df_all_mice[inhibitory_neurons & watson_result_exists])))
    watson_significant = df_all_mice.watson_test_hd > 0.385  # p < 0.001
    print('Number of cells with significantly differen HD distributions: ' + str(watson_significant.sum()))
    print('Number of excitatory neurons with significantly different HD: ' + str(len(df_all_mice[watson_significant & excitatory_neurons])))
    print('Number of inhibitory neurons with significantly different HD: ' + str(len(df_all_mice[watson_significant & inhibitory_neurons])))

    print('Number of excitatory neurons with significantly different HD that are hd cells: ' + str(len(df_all_mice[watson_significant & excitatory_neurons & is_hd_cell])))
    print('Number of inhibitory neurons with significantly different HD that are hd cells: ' + str(len(df_all_mice[watson_significant & inhibitory_neurons & is_hd_cell])))

    # find those where the hd polar plot significantly differs in the first vs second half
    # run load_df again and add the result of the 2 sample watson test to the big df and then filter for that here
    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_mice.hd_correlation_first_vs_second_half[significant_corr & good_cluster & watson_significant], save_output_path + 'correlation_hd_session.png')
    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_mice.hd_correlation_first_vs_second_half[significant_corr & good_cluster & watson_significant & excitatory_neurons], save_output_path + 'correlation_hd_session_excitatory.png')
    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_mice.hd_correlation_first_vs_second_half[significant_corr & good_cluster & watson_significant & inhibitory_neurons], save_output_path + 'correlation_hd_session_inhibitory.png')


if __name__ == '__main__':
    main()
