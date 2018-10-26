import pandas as pd
import PostSorting.open_field_head_direction
import numpy as np
import data_frame_utility


# source: https://stackoverflow.com/users/48956/user48956
def df_empty(columns, dtypes, index=None):
    assert len(columns) == len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def get_field_data_frame(spatial_firing):
    #print(spatial_firing.head())
    field_data = data_frame_utility.df_empty(['session_id', 'cluster_id', 'field_id', 'indices_rate_map', 'spike_times', 'position_x_spikes', 'position_y_spikes', 'hd_in_field_spikes', 'hd_hist', ])
    for cluster in range(len(spatial_firing)):
        number_of_spikes_in_fields = []
        number_of_samples_in_fields = []
        hd_in_fields_cluster = []
        hd_in_field_sessions = []
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        number_of_firing_fields = len(spatial_firing.firing_fields[cluster])
        if number_of_firing_fields > 0:
            firing_field_spike_times = spatial_firing.spike_times_in_fields[cluster]
            firing_field_times_session = spatial_firing.times_in_session_fields[cluster]
            for field_id, field in enumerate(firing_field_spike_times):
                mask_firing_times_in_field = np.in1d(spatial_firing.firing_times[cluster], field)
                firing_field_times_session = spatial_firing.times_in_session_fields[cluster][field_id]
                number_of_spikes_field = len(field)
                hd_field_cluster = np.array(spatial_firing.hd[cluster])[mask_firing_times_in_field]
                hd_field_cluster = (np.array(hd_field_cluster) + 180) * np.pi / 180
                hd_fields_cluster_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_field_cluster)
                number_of_spikes_in_fields.append(number_of_spikes_field)
                hd_in_fields_cluster.append(hd_fields_cluster_hist)



def main():
    spatial_firing = pd.read_pickle('C:/Users/s1466507\Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of/DataFrames/spatial_firing.pkl')
    get_field_data_frame(spatial_firing)

if __name__ == '__main__':
    main()