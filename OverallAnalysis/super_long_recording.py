import data_frame_utility
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_field_analysis
import pandas as pd
import PostSorting.parameters

import scipy


local_path = OverallAnalysis.folder_path_settings.get_local_path()
analysis_path = local_path + '/super_long_recording/'

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


def get_number_of_directional_fields(fields, tag='grid'):
    percentiles_no_correction = []
    percentiles_correction = []
    for index, field in fields.iterrows():
        percentile = scipy.stats.percentileofscore(field.number_of_different_bins_shuffled, field.number_of_different_bins)
        percentiles_no_correction.append(percentile)

        percentile = scipy.stats.percentileofscore(field.number_of_different_bins_shuffled_corrected_p, field.number_of_different_bins_bh)
        percentiles_correction.append(percentile)

    print(tag)
    print('Number of fields: ' + str(len(fields)))
    print('Number of directional fields [without correction]: ')
    print(np.sum(np.array(percentiles_no_correction) > 95))
    fields['directional_no_correction'] = np.array(percentiles_no_correction) > 95

    print('Number of directional fields [with BH correction]: ')
    print(np.sum(np.array(percentiles_correction) > 95))
    fields['directional_correction'] = np.array(percentiles_correction) > 95
    print('Percentile values, with correction:')
    print(percentiles_correction)


def process_data():
    # load shuffled field data
    if os.path.exists(analysis_path + 'DataFrames/fields.pkl'):
        shuffled_fields = pd.read_pickle(analysis_path + 'DataFrames/fields.pkl')
    else:
        firing = pd.read_pickle(analysis_path + 'DataFrames/spatial_firing.pkl')
        position = pd.read_pickle(analysis_path + 'DataFrames/position.pkl')
        shuffled_fields = get_shuffled_field_data(firing, position)
        shuffled_fields.to_pickle(analysis_path + 'DataFrames/fields.pkl')

    number_of_significant_bins = shuffled_fields.number_of_different_bins_bh
    print(number_of_significant_bins)
    get_number_of_directional_fields(shuffled_fields, tag='grid')


def main():
    process_data()


if __name__ == '__main__':
    main()
