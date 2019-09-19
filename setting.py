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
movement_ch = '100_ADC2.continuous' #channel that contains the movement data
opto_ch = '100_ADC3.continuous'
data_file_prefix = '100_CH' #prefix of data files
data_file_suffix = '' 
wave_form_size = 40 

#########
# sorter configuration
sorterName = 'mountainsort4'
is_tetrode_by_tetrode = False #set to True if you want the spike sorting to be done tetrode by tetrode
all_tetrode_together = True #set to True if you want the spike sorting done on all tetrodes combined


##########
# VR
track_length = 200
first_trial_channel = '100_ADC4.continuous'
second_trial_channel = '100_ADC5.continuous'


##########
# Experiment
session_type = 'vr'


##########
# open field
opto_tagging_start_index = None

