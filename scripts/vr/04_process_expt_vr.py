#%%
import PostSorting.vr_spatial_firing
import PostSorting.vr_firing_rate_maps
import PostSorting.vr_FiringMaps_InTime
import setting
import pandas as pd
from collections import namedtuple
from types import SimpleNamespace
import SnakeIOHelper
import scipy.signal as signal

# #%% define sinput and output
(sinput, soutput) = SnakeIOHelper.getSnake(locals(), 'workflow/workflow_vr.smk', [setting.debug_folder+'/processed/spatial_firing_vr.pkl'],
    'process_expt')

#%% Load data
spike_data = pd.read_pickle(sinput.spatial_firing)
raw_position_data =pd.read_pickle(sinput.raw_position)
processed_position_data = pd.read_pickle(sinput.processed_position_data)

#%% process firing times
downsample_ratio = setting.sampling_rate / setting.location_ds_rate
_, _, spike_data = PostSorting.vr_spatial_firing.process_spatial_firing(spike_data, raw_position_data, downsample_ratio)
spike_data_vr = PostSorting.vr_spatial_firing.split_spatial_firing_by_trial_type(spike_data)
spike_data_vr = PostSorting.vr_firing_rate_maps.make_firing_field_maps(spike_data, processed_position_data, 
    setting.track_length/setting.location_bin_num, setting.track_length)

#%% save data
spike_data_vr.to_pickle(soutput.spatial_firing_vr)

#%%
# #%%
# import numpy as np 
# chIdx = 1
# cluster_trial_x_locations = spike_data_vr.x_position_cm[chIdx]
# bins = np.arange(0,200)
# beaconed_bin_counts = np.digitize(cluster_trial_x_locations, bins)
# np.sum(beaconed_bin_counts==184)
# # %%
# st = spike_data_vr.firing_times[1]
# bt = np.arange(st[0], st[-1], 30000/10)
# st_hist = np.histogram(st,bt)
# plt.plot(st_hist[1][1:], st_hist[0])

# # %%
# pos = raw_position_data.x_position_cm
# plt.hist(pos,200)

# %% Plot per channel
# cluster_spike_data = spike_data_vr[(spike_data_vr["cluster_id"] == 3)]
# x_locations_cm = np.array(cluster_spike_data["x_position_cm"].tolist()[0])
# trial_types = np.array(cluster_spike_data["trial_type"].tolist()[0])
# spike_locations = x_locations_cm[trial_types == 0]
# trial_numbers = np.array(cluster_spike_data["trial_number"].tolist()[0])
# spike_trial_numbers = trial_numbers[trial_types == 0]
# bins = np.arange(0,200)
# total_count =[]
# for trial_number in processed_position_data["trial_number"].unique():
# # for trial_number in [1]:
#     trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial_number]
#     trial_spike_locations = spike_locations[spike_trial_numbers==trial_number]
#     trial_binned_time = trial_processed_position_data["times_binned"].iloc[0]
#     spike_bin_counts = np.histogram(trial_spike_locations, bins)[0]
#     normalised_rate_map = spike_bin_counts/trial_binned_time
#     print(spike_bin_counts[183])
#     total_count.append(spike_bin_counts)
#     # print(trial_binned_time)
#     # print(trial_spike_locations)
#     # plt.figure()
#     # plt.plot(normalised_rate_map)
#     # plt.title(trial_number)

# total_count = np.array(total_count).sum(0)
# print(total_count)

# #%%
# cluster_spike_data = spike_data_vr[(spike_data_vr["cluster_id"] == 3)]
# trial_types = np.array(cluster_spike_data["trial_type"].tolist()[0])
# x_locations_cm = np.array(cluster_spike_data["x_position_cm"].tolist()[0])
# trial_numbers = np.array(cluster_spike_data["trial_number"].tolist()[0])
# x_b = x_locations_cm[trial_types==0]
# x_trial_number = trial_numbers[trial_types==0]

# h1 = np.histogram(x_b,bins)
# print(h1[0][183])
# print('------')
# for i in range(0,20):
#     h = np.histogram(x_b[x_trial_number==i],bins)[0]
#     print(h[183])

# # if a particular trial is not included in the trial type, and the animal has stay for a long time in that trial, the 
# # dwell time may be wrong

# # %%
# trialNo=13
# pos_t = np.arange(len(raw_position_data.x_position_cm))*30
# pos = raw_position_data.x_position_cm[raw_position_data.trial_number ==trialNo]

# # smooth the position data
# b,a = signal.butter(5,10/(1000/2))
# pos_s = signal.filtfilt(b,a, pos)

# bins = np.arange(0,200)
# pos_d  = np.digitize(pos,bins)

# pos_tt = pos_t[raw_position_data.trial_number ==trialNo]
# sk = spike_data.firing_times.iloc[1][np.array(spike_data.trial_number.iloc[1]) ==trialNo]
# fig,ax = plt.subplots(3,1,sharex='col')


# ax[0].plot(pos_tt,pos)
# ax[0].set_title('Raw position data')
# ax[0].set_ylim([184,189])

# ax[1].plot(pos_tt, pos_d,'.')
# ax[1].set_title('Binned position data')
# ax[1].set_ylim([184,190])

# ax[2].eventplot(sk)
# ax[2].set_title('Spikes')
# x_range = np.where((pos>184) & (pos<188))[0]
# ax[0].set_xlim([pos_tt[x_range[0]],pos_tt[x_range[-1]]])
# fig.tight_layout()
# # %%
