import numpy as np
import pandas as pd
import matplotlib.pylab as plt

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


def main():
    shuffled_field_data = pd.read_pickle(local_path_to_shuffled_field_data)
    shuffled_field_data = get_accepted_fields(shuffled_field_data)
    number_of_rejects = shuffled_field_data.number_of_different_bins
    plt.hist(number_of_rejects)
    pass
    # find 95th percentile and see how many fields have a significant number of rejects


if __name__ == '__main__':
    main()
