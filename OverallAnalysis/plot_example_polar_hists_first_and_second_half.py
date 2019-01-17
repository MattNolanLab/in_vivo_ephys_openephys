import pandas as pd
import OverallAnalysis.false_positives
import OverallAnalysis.analyze_field_correlations
import PostSorting.open_field_make_plots


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
    local_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df.pkl'
    path_to_data = 'C:/Users/s1466507/Documents/Ephys/recordings/'
    save_output_path = 'C:/Users/s1466507/Documents/Ephys/overall_figures/'
    false_positives_path = path_to_data + 'false_positives_all.txt'
    df_all_mice = load_data_frame(local_path)
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(false_positives_path)
    df_all_mice = add_combined_id_to_df(df_all_mice)
    df_all_mice['false_positive'] = df_all_mice['false_positive_id'].isin(list_of_false_positives)

    good_cluster = df_all_mice.false_positive == False

    first_half_histogram_excitatory = df_all_mice.loc[df_all_mice['session_id'] == 'M12_2018-04-03_13-11-02_of'].hd_hist_first_half[9]
    second_half_histogram_excitatory = df_all_mice.loc[df_all_mice['session_id'] == 'M12_2018-04-03_13-11-02_of'].hd_hist_second_half[9]

    first_half_histogram_inhibitory = df_all_mice.loc[df_all_mice['session_id'] == 'M13_2018-04-27_10-29-02_of'].hd_hist_first_half[15]
    second_half_histogram_inhibitory = df_all_mice.loc[df_all_mice['session_id'] == 'M13_2018-04-27_10-29-02_of'].hd_hist_second_half[15]

    PostSorting.open_field_make_plots.plot_polar_hd_hist(first_half_histogram_excitatory, second_half_histogram_excitatory, 9, save_output_path + 'first_vs_second_half_M12_2018-04-03_13-11-02_of')
    PostSorting.open_field_make_plots.plot_polar_hd_hist(first_half_histogram_inhibitory, second_half_histogram_inhibitory, 9, save_output_path + 'first_vs_second_half_M13_2018-04-27_10-29-02_of')


if __name__ == '__main__':
    main()
