import data_frame_utility
import os
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_field_analysis
import pandas as pd
import PostSorting.parameters


local_path = OverallAnalysis.folder_path_settings.get_local_path()
analysis_path = local_path + '/compare_directional_firing_over_days/'

prm = PostSorting.parameters.Parameters()
prm.set_pixel_ratio(440)


def get_shuffled_field_data(spatial_firing, position_data, shuffle_type='distributive', sampling_rate_video=50):
    field_df = data_frame_utility.get_field_data_frame(spatial_firing, position_data)
    field_df = OverallAnalysis.shuffle_field_analysis.add_rate_map_values_to_field_df_session(spatial_firing, field_df)
    field_df = OverallAnalysis.shuffle_field_analysis.shuffle_field_data(field_df, analysis_path, number_of_bins=20,
                                  number_of_times_to_shuffle=1000, shuffle_type=shuffle_type)
    field_df = OverallAnalysis.shuffle_field_analysis.analyze_shuffled_data(field_df, analysis_path, sampling_rate_video,
                                     number_of_bins=20, shuffle_type=shuffle_type)
    return field_df


def process_data():
    # load shuffled field data
    if os.path.exists(analysis_path + 'DataFrames_1/fields.pkl'):
        fields1 = pd.read_pickle(analysis_path + 'DataFrames_1/fields.pkl')
    else:
        day1_firing = pd.read_pickle(analysis_path + 'DataFrames_1/spatial_firing.pkl')
        day1_position = pd.read_pickle(analysis_path + 'DataFrames_1/position.pkl')
        shuffled_fields_1 = get_shuffled_field_data(day1_firing, day1_position)
        shuffled_fields_1.to_pickle(analysis_path + 'DataFrames_1/fields.pkl')

    if os.path.exists(analysis_path + 'DataFrames_2/fields.pkl'):
        fields2 = pd.read_pickle(analysis_path + 'DataFrames_1/fields.pkl')
    else:
        day2_firing = pd.read_pickle(analysis_path + 'DataFrames_2/spatial_firing.pkl')
        day2_position = pd.read_pickle(analysis_path + 'DataFrames_2/position.pkl')
        # shuffle field analysis
        shuffled_fields_2 = get_shuffled_field_data(day2_firing, day2_position)
        shuffled_fields_2.to_pickle(analysis_path + 'DataFrames_2/fields.pkl')
        print('I shuffled data from both days.')


    # plot them both
    # identity of directional bins?



def main():
    process_data()


if __name__ == '__main__':
    main()
