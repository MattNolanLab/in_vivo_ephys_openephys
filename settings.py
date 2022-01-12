#This file contains the parameters for analysis


##########
# Folder locations
basefolder = '/Users/teristam/Documents/Data'
mountainsort_tmp_folder = basefolder+'/tmp/mountainlab/'
sorting_folder = basefolder+'/to_sort/recordings/'
to_sort_folder = basefolder+'/to_sort/'
output_folder = basefolder +'/output'
server_path_first_half = basefolder+'/log/'
downtime_lists_path = basefolder+'/to_sort/sort_downtime/'

##########
# Recording setting
sampling_rate = 30000
num_tetrodes = 4
movement_ch_suffix = f'ADC2' #channel that contains the movement data
opto_ch_suffix = f'ADC3'
data_file_prefix = f'_CH' #prefix of data files
data_file_suffix = '' 
wave_form_size = 40 
tetrodeNum = 4 #how many channel in one tetrode

#########
# sorter configuration
sorterName = 'mountainsort4'
is_tetrode_by_tetrode = False #set to True if you want the spike sorting to be done tetrode by tetrode
all_tetrode_together = True #set to True if you want the spike sorting done on all tetrodes combined


############
# Analysis
spike_bin_size = 20 #the bin size to group spike together to calculate spike count, in ms
stop_threshold = 4.7 #threshold for detecting stop
location_ds_rate = 1000 #the sampleing frequency in Hz to downsample the location signal to 

##########
# VR
track_length = 200
first_trial_channel_suffix = f'ADC4' #channel for the start of trial
second_trial_channel_suffix = f'ADC5' #channel for the stp of trial
reward_start = 88 #position for the reward
reward_end = 110 #position for the reward
vr_bin_size_cm = 1
movement_threshold = 2.5
goal_location_chennl_suffix=f'ADC7'


##########
# Experiment
session_type = 'vr'


##########
# open field
opto_tagging_start_index = None
pixel_ratio = 440
sync_channel_suffix = 'ADC1' #channel for the sync pulse
bonsai_sampling_rate = 30

###################
# Binning
theta_bin = 18
position_bin = 100
speed_bin = 20
accel_bin = 20 
binSize = 100 # in ms
trackLength = 200 # TODO: should load this from the parameter file instead

###################
# LFP
lfp_lp = 400 # low pass filter in Hz
lfp_hp = 10 # high pass filter
lfp_fs = 1000 # sampling frequency of the lfp signal

############
# Debug
# debug_folder = '/mnt/datastore/Teris/FragileX/cohort_202108/data/openfield/M3_D27_2021-09-16_12-59-00'
# debug_folder = '/mnt/datastore/Teris/FragileX/cohort_202108/data/VR/M3_D36_2021-10-11_12-36-06'
debug_folder = '/mnt/datastore/Sarah/Data/OptoEphys_in_VR/Data/OpenEphys/_cohort3/VirtualReality/M1_D20_2018-10-15_11-51-45'