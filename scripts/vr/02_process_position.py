#%%
import pandas as pd
from PostSorting.vr_sync_spatial_data import *
import PostSorting.vr_spatial_data as vr_spatial_data
import PostSorting.vr_speed_analysis as vr_speed_analysis
import PostSorting.vr_time_analysis as vr_time_analysis
import PostSorting.vr_stop_analysis as vr_stop_analysis
import PostSorting.vr_make_plots as vr_make_plots
import PostSorting
from collections import namedtuple
import settings
from types import SimpleNamespace
import gc
import PostSorting.vr_stop_analysis as vr_stop_analysis
import scipy.signal as signal
from utils import SnakeIOHelper
#%% Define input and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr.smk', [settings.debug_folder+'/processed/processed_position.pkl'],
    'process_position')

#%% Load and downsample the position data
print('Loading location and trial onset files')
recorded_location = get_raw_location(sinput.recording_to_sort, soutput.raw_position_plot) # get raw location from DAQ pin
first_ch = load_first_trial_channel(sinput.recording_to_sort)
second_ch = load_second_trial_channel(sinput.recording_to_sort)

print('Downsampling')
ds_ratio = int(settings.sampling_rate/settings.location_ds_rate)
recorded_location_ds=downsample(recorded_location, ds_ratio) #filtering may cause
first_ch_ds = downsample(first_ch, ds_ratio)
second_ch_ds = downsample(second_ch, ds_ratio)

#%% Process position data
raw_position_data = pd.DataFrame()
raw_position_data = calculate_track_location(raw_position_data, recorded_location_ds, settings.track_length)
raw_position_data = calculate_trial_numbers(raw_position_data, soutput.trial_figure)

#%% Calculate trial-related information
raw_position_data = calculate_trial_types(raw_position_data, first_ch_ds, second_ch_ds, soutput.trial_type_plot_folder)
raw_position_data = calculate_time(raw_position_data, settings.location_ds_rate)
raw_position_data = calculate_instant_dwell_time(raw_position_data, settings.location_ds_rate)
raw_position_data = calculate_instant_velocity(raw_position_data, soutput.speed_plot, 
            settings.location_ds_rate, speed_win = 0.1)
raw_position_data = get_avg_speed_200ms(raw_position_data, soutput.mean_speed_plot, settings.location_ds_rate,0.1)

#%% save data
raw_position_data.to_pickle(soutput.raw_position_data)

#%% bin the position data over trials
processed_position_data = pd.DataFrame() # make dataframe for processed position data
processed_position_data = vr_speed_analysis.calculate_binned_speed(raw_position_data,processed_position_data, settings.track_length)
processed_position_data = vr_time_analysis.calculate_binned_time(raw_position_data,processed_position_data,settings.track_length)


#%% Analysis stops
#TODO: load stop threshold from parameter file
processed_position_data = vr_stop_analysis.get_stops_from_binned_speed(processed_position_data, 4.7)
processed_position_data = vr_stop_analysis.calculate_average_stops(processed_position_data)
processed_position_data = vr_stop_analysis.calculate_first_stops(processed_position_data)
processed_position_data = vr_stop_analysis.calculate_rewarded_stops(processed_position_data)
processed_position_data =vr_stop_analysis.calculate_rewarded_trials(processed_position_data)


#%% plotting position data
vr_make_plots.plot_stops_on_track(processed_position_data, soutput.stop_raster)
vr_make_plots.plot_stop_histogram(processed_position_data, soutput.stop_histogram)
vr_make_plots.plot_speed_histogram(processed_position_data, soutput.speed_histogram)
vr_make_plots.plot_speed_per_trial(processed_position_data, soutput.speed_heat_map, track_length=settings.track_length)

#%% save data
processed_position_data.to_pickle(soutput.processed_position_data)

# # %%

# path = '/mnt/datastore/Teris/FragileX/data/VR/M2_D28_2021-02-17_15-53-49/Teris2021_20210217_1554.csv'
# headers = ['time','position','speed','speed_over_gain','reward_is_received','reward_is_failed','lick_detected',
#     'tone_played','track_idx','tot_trials','gain_mod','rz_start','rz_stop', 'sync_pulse']
# df_blender = pd.read_csv(path,skiprows=4,sep=';',names=headers)
# df_blender

# # %%
# from scipy import interpolate
# ephys_pos = raw_position_data.x_position_cm.values
# ephy_spd = raw_position_data.speed_per200ms.values

# blender_pos = df_blender.position.values
# blender_speed = df_blender.speed.values
# blender_time = df_blender.time.values
# t = np.arange(blender_time[0],blender_time[-1],0.03333)
# # resample the blender pos
# f = interpolate.interp1d(blender_time,blender_pos)
# blender_pos2 = f(t)
# f_spd = interpolate.interp1d(blender_time,blender_speed)
# blender_speed2 = f_spd(t)

# #%%
# corr = np.correlate(blender_pos2[:30*60*5], ephys_pos[:30*60*5],'full')
# # try to align the two data together
# lag = int((np.argmax(corr) - (corr.size + 1)/2))  # lag between sync pulses is based on max correlation
# print(lag)
# plt.plot(corr)
# ephys_pos_aligned = ephys_pos[-lag:]
# ephys_spd_aligned = ephy_spd[-lag:]
# # %%
# fig,ax = plt.subplots(4,1,sharex=True,figsize=(12,10))

# ax[0].plot((blender_speed2<4.7) & (blender_pos2>8.8) & (blender_pos2<11) ,'red')
# ax[0].set_title('Blender stop in reward')

# ax[1].plot((ephys_spd_aligned<4.7) & (ephys_pos_aligned>88) & (ephys_pos_aligned<110),'blue')
# ax[1].set_title('Ephys stop in reward')

# ax[2].plot(blender_pos2*10,'red',label='blender')
# ax[2].plot(ephys_pos_aligned,'blue', label='ephys')
# ax[2].set_title('Position')

# ax[3].plot(blender_speed2,'red',label='blender')
# ax[3].plot(ephys_spd_aligned,'blue', label='ephys')
# ax[3].set_title('Speed')


# ax[0].set_xlim([3500,30*60*2.5])
# # %%
# path = '/mnt/datastore/Teris/FragileX/data/VR/M2_D28_2021-02-17_15-53-49/MountainSort/DataFrames/processed_position_data.pkl'
# df_old = pd.read_pickle(path)
# # %%

# %%
