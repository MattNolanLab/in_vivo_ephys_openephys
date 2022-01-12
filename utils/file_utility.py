import glob
import os
import settings
from utils import OpenEphys
import numpy as np
import settings
import glob
import pickle
import logging

logger = logging.getLogger(__name__)

def load_recording_info(recording_info):
    with open(recording_info,'rb') as f:
        d = pickle.load(f)

    return d['serialized_dict']['kwargs']

def search4File(fname, expected_num_file=1):
    file = glob.glob(fname)
    if not len(file) == expected_num_file:
        raise FileNotFoundError(f'Error, cannot find exact files, candidate: {file}')
    
    return file[0]

def load_openephys_file(fname,search4match=True, auto_concat=True):
    '''
    Load an openephys file, taking into account the possible file split and different suffix e.g. ADC1_2 ADC1_3 etc.
    '''
    j = 2
    x = None
    try:
        fn = search4File(fname+'.continuous') #search for the data file
        x = OpenEphys.loadContinuousFast(fn)['data']
    except FileNotFoundError:
        # the default file name not found, see if it is saved as _2 or _3 etc 
        while j<10:
            try:
                fn = search4File(fname+f'_{j}.continuous') #search for the data file
                x = OpenEphys.loadContinuousFast(fn)['data']
                j += 1
                break
            except:
                j += 1
        
        if j >= 10:
            raise FileNotFoundError(f'Error, cannot find the data files')
        
    if auto_concat:
        while True:
            # Also try to search for other files
            try:
                fn = search4File(fname+f'_{j}.continuous') #search for the data file
                x2 = OpenEphys.loadContinuousFast(fn)['data']
                print('Another data file found. Concatenating')
                x = np.concatenate([x,x2])
                j += 1
            except FileNotFoundError:
                break

    assert x is not None, f'{fname}'
    return x


def load_OpenEphysRecording(folder, auto_concat=True, correct_data_length=True):
    '''
    auto_concat: automatically concat two consecutive recordings produced by openephys together e.g. 101_ADC1, 101_ADC1_2
    correct_data_length: whether to trunciate to the shortest data file if the data length of the files are not the same
    '''

    signal = []
    for i in range(settings.num_tetrodes*4):
        # Different open ephys setting may lead to different source id for the data file
        # so search for the data file instead of using hard-coded name

        x = load_openephys_file(folder+'/*_CH'+str(i+1))

        if i==0:
            #preallocate array on first run
            signal = np.zeros((settings.num_tetrodes*4,x.shape[0]))

        if correct_data_length:
            # attempt to correct for signal length difference
            if len(x) > signal.shape[1]:
                signal[i,:] = x[:signal.shape[1]]
                print('Warning: Channel data length not matched. Some data are truncated')
            elif len(x) < signal.shape[1]:
                signal = signal[:,:len(x)]
                signal[i,:] = x
                print('Warning: Channel data length not matched. Some data are truncated')
            else:
                signal[i,:] = x
        else:
            signal[i,:] = x
    return signal

def getDeadChannel(deadChannelFile):
    with open(deadChannelFile,'r') as f:
        deadChannels = [int(s) for s in f.readlines()]
    
    return deadChannels

def find_the_file(file_path, pattern, type):
    name = None
    file_found = True
    file_name = None

    file_counter = 0
    for name in glob.glob(file_path + pattern):
        file_counter += 1
        pass

    if file_counter > 1:
        print('There are more than one ' + type + ' files in this folder. This may not be okay.')

    if name is not None:
        file_name = name.rsplit('\\', 1)[1]
    else:
        print('The '+ type + ' file(such as ' + pattern + ' )is not here, or it has an unusual name.')

        file_found = False

    return file_name, file_found


def init_data_file_names(prm, beginning, end):
    prm.set_continuous_file_name(beginning)
    prm.set_continuous_file_name_end(end)


def set_continuous_data_path(prm):
    file_path = prm.get_filepath()
    continuous_file_name_1 = '105_CH'
    continuous_file_name_end_1 = '_0'
    continuous_file_name_2 = '100_CH'
    continuous_file_name_end_2 = ''

    recording_path = file_path + continuous_file_name_1 + str(1) + continuous_file_name_end_1 + '.continuous'
    if os.path.isfile(recording_path) is True:
        init_data_file_names(prm, continuous_file_name_1, continuous_file_name_end_1)

    recording_path = file_path + continuous_file_name_2 + str(1) + continuous_file_name_end_2 + '.continuous'
    if os.path.isfile(recording_path) is True:
        init_data_file_names(prm, continuous_file_name_2, continuous_file_name_end_2)

def set_dead_channel_path(prm):
    file_path = prm.get_filepath()
    dead_ch_path = file_path + "/dead_channels.txt"
    prm.set_dead_channel_path(dead_ch_path)


def create_behaviour_folder_structure(prm):
    movement_path = prm.get_filepath() + 'Behaviour'
    prm.set_behaviour_path(movement_path)

    data_path = movement_path + '/Data'
    analysis_path = movement_path + '/Analysis'

    prm.set_behaviour_data_path(data_path)
    prm.set_behaviour_analysis_path(analysis_path)

    if os.path.exists(movement_path) is False:
        print('Behavioural data will be saved in {}.'.format(movement_path))
        os.makedirs(movement_path)
        os.makedirs(data_path)
        os.makedirs(analysis_path)


# main path is the folder that contains 'recordings' and 'sorting_files'
def get_main_path(prm):
    file_path = prm.get_filepath()
    main_path = file_path.rsplit('/', 3)[-4]
    return main_path


def get_raw_mda_path_all_channels(prm):
    raw_mda_path = prm.get_filepath() + 'Electrophysiology/' + prm.get_spike_sorter() + '/raw.mda'
    return raw_mda_path


def get_raw_mda_path_separate_tetrodes(prm):
    raw_mda_path = '/data/raw.mda'
    return raw_mda_path


def folders_for_separate_tetrodes(prm):
    ephys_path = prm.get_filepath() + 'Electrophysiology'

    spike_path = ephys_path + '/Spike_sorting'
    data_path = ephys_path + '/Data'
    sorting_t1_path_continuous = spike_path + '/t1'
    sorting_t2_path_continuous = spike_path + '/t2'
    sorting_t3_path_continuous = spike_path + '/t3'
    sorting_t4_path_continuous = spike_path + '/t4'

    mountain_data_folder_t1 = spike_path + '/t1/data'
    mountain_data_folder_t2 = spike_path + '/t2/data'
    mountain_data_folder_t3 = spike_path + '/t3/data'
    mountain_data_folder_t4 = spike_path + '/t4/data'

    if os.path.exists(ephys_path) is False:
        os.makedirs(ephys_path)
        os.makedirs(spike_path)
        os.makedirs(data_path)

    if os.path.exists(sorting_t1_path_continuous) is False:
        os.makedirs(sorting_t1_path_continuous)
        os.makedirs(sorting_t2_path_continuous)
        os.makedirs(sorting_t3_path_continuous)
        os.makedirs(sorting_t4_path_continuous)

        os.makedirs(mountain_data_folder_t1)
        os.makedirs(mountain_data_folder_t2)
        os.makedirs(mountain_data_folder_t3)
        os.makedirs(mountain_data_folder_t4)


def create_ephys_folder_structure(prm):
    ephys_path = prm.get_filepath() + 'Electrophysiology'
    prm.set_ephys_path(ephys_path)
    data_path = ephys_path + '/' + prm.get_spike_sorter()

    if os.path.exists(ephys_path) is False:
        os.makedirs(ephys_path)
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)


def create_folder_structure(prm):
    create_behaviour_folder_structure(prm)
    create_ephys_folder_structure(prm)


def convertContinuous2Binary(continuousFolder, binaryFolder):
    """Convert continuous files to float binary format
    
    Arguments:
        continuousFolder {str} -- folder of continuous files
        binaryFolder {str} -- target folder of flat binary format
    """

    data,headers = OpenEphys.load_OpenEphysRecording4BinaryFile(continuousFolder)
    OpenEphys.writeBinaryData(binaryFolder,data)
    OpenEphys.writeStructFile(binaryFolder+'/structure.oebin',headers)



def get_location_on_server(recording_directory):
    parameters_path = recording_directory + '/parameters.txt'
    param_file_reader = open(parameters_path, 'r')
    parameters = param_file_reader.readlines()
    parameters = list([x.strip() for x in parameters])
    location_on_server = parameters[1]
    return location_on_server

def parse_parameter_file(parameter_file_path):
    # parse the parameter file
    parameters_path = parameter_file_path
    lines = open(parameters_path,'r').read().split('\n')
    d = {}
    d['expt_type'] = lines[0]
    d['path'] = lines[1]
    
    if len(lines) > 2:
        d['tag'] = lines[2].split()
    else:
        d['tag'] = []
    
    return d


def get_tags_parameter_file(recording_directory):
    tags = False
    try:
        parameters_path = recording_directory + '/parameters.txt'
        param_file_reader = open(parameters_path, 'r')
        parameters = param_file_reader.readlines()
        parameters = list([x.strip() for x in parameters])
        if len(parameters) > 2:
            tags = parameters[2]
    finally:
        return tags


def write_param_file_for_matlab(file_to_sort, path_to_server, is_openfield, is_vr, matlab_params_file_path, server_path_first_half):
    if is_openfield:
        openfield = 1
    else:
        openfield = 0
    opto = 1
    params_for_matlab_file = open(matlab_params_file_path + "PostClusteringParams.txt", "w")
    params_for_matlab_file.write(file_to_sort + ',\n')
    params_for_matlab_file.write(server_path_first_half + path_to_server + ',\n')
    params_for_matlab_file.write(str(openfield) + ',\n')
    params_for_matlab_file.write(str(opto))
    params_for_matlab_file.close()


def write_shell_script_to_call_matlab(file_to_sort):
    script_path = file_to_sort + '/run_matlab.sh'
    batch_writer = open(script_path, 'w', newline='\n')
    batch_writer.write('#!/bin/bash\n')
    batch_writer.write('echo "-----------------------------------------------------------------------------------"\n')
    batch_writer.write('echo "This is a shell script that will call matlab."\n')
    batch_writer.write('export MATLABPATH=/home/nolanlab/PycharmProjects/in_vivo_ephys_openephys/PostClustering/\n')

    batch_writer.write('matlab -r PostClusteringAuto')

def get_track_info_log_file(recording_path, session_config_path):
    '''
    Read track length and reward location from
    '''
    logger.info('Reading track info from log file')

    logfile_path = glob.glob(os.path.join(recording_path,'*.log'))

    if len(logfile_path) == 0:
        raise FileNotFoundError("Cannot found the log file. Have you copied it to the recording folder?")
    else:
        logfile_path = logfile_path[0]
    

    with open(logfile_path, 'r') as f:
        log_file = f.readlines()
    
    # search for the line specifying the config file
    for l in log_file:
        if 'Using config fle' in l:
            session_file_name = l.split()[-1]
            break
    
    # Open the session config file and load the track info
    with open(os.path.join(session_config_path, session_file_name)) as f:
        first_trial = f.readlines()[1].split(';')
        reset = float(first_trial[9])*10 #conver to cm
        reward_zone_start = float(first_trial[11])*10
    
    return reset, reward_zone_start, session_file_name