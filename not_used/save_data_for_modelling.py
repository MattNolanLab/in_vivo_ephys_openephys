import pandas as pd


spatial_data_local_path ='/Users/s1466507/Documents/Ephys/recordings/M14_2018-05-16_11-29-05_of/MountainSort/DataFrames/spatial_firing.pkl'
position_local_path = '/Users/s1466507/Documents/Ephys/recordings/M14_2018-05-16_11-29-05_of/MountainSort/DataFrames/position.pkl'
cluster_id = 12

spatial_data = pd.read_pickle(spatial_data_local_path)
position_data = pd.read_pickle(position_local_path)

position_data.to_csv(position_local_path + '.csv')

spike_df = pd.DataFrame()

position_x_spike = spatial_data.position_x[cluster_id]
position_y_spike = spatial_data.position_y[cluster_id]
position_x_spike_pixels = spatial_data.position_x_pixels[cluster_id]
position_y_spike_pixels = spatial_data.position_y_pixels[cluster_id]
hd_spike = spatial_data.hd[cluster_id]
spike_df['position_x'] = position_x_spike
spike_df['position_y'] = position_y_spike
spike_df['position_x_pixels'] = position_x_spike_pixels
spike_df['position_y_pixels'] = position_y_spike_pixels

spike_df.to_csv(spatial_data_local_path + '.csv')
pass