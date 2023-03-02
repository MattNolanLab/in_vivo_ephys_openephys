##some helper functions to be used with spike interface
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface.sorters as sorters
import spikeinterface.preprocessing as spre
import OpenEphys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt
import pandas as pd
from file_utility import *
import settings
from tqdm import tqdm
from probeinterface import Probe
from probeinterface.plotting import plot_probe
from probeinterface import get_probe
from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe_group
import data_frame_utility
import tempfile

def load_OpenEphysRecording(folder, channel_ids=None):
    number_of_channels, corrected_data_file_suffix = count_files_that_match_in_folder(folder, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')
    if channel_ids is None:
        channel_ids = np.arange(1, number_of_channels+1)

    signal = []
    for i, channel_id in enumerate(channel_ids):
        fname = folder+'/'+corrected_data_file_suffix+str(channel_id)+settings.data_file_suffix+'.continuous'
        x = OpenEphys.loadContinuousFast(fname)['data']
        if i==0:
            #preallocate array on first run
            signal = np.zeros((x.shape[0], len(channel_ids)))
        signal[:,i] = x
    return [signal]


def test_probe_interface(save_path):
    num_channels = 64
    recording, sorting_true = se.toy_example(duration=5, num_channels=num_channels, seed=0, num_segments=4)

    other_probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
    print(other_probe)

    other_probe.set_device_channel_indices(np.arange(num_channels))
    recording_4_shanks = recording.set_probe(other_probe, group_mode='by_shank')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    plot_probe(recording_4_shanks.get_probe(), ax=ax)
    plt.savefig(save_path+'probe_locations.png', dpi=200)
    plt.close()

def add_probe_info(recordingExtractor, recording_to_sort, sorterName):
    number_of_channels, corrected_data_file_suffix = count_files_that_match_in_folder(recording_to_sort, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')

    if number_of_channels == 16: # presume tetrodes
        geom = pd.read_csv(settings.tetrode_geom_path, header=None).values
        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': 5})
        probe.set_device_channel_indices(np.arange(number_of_channels))
        recordingExtractor.set_probe(probe, in_place=True)
    else: # presume cambridge neurotech P1 probes

        n_probes = number_of_channels/64
        probegroup =ProbeGroup()
        for i in range(int(n_probes)):
            probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
            probe.move([i*2000, 0]) # move the probes far away from eachother
            probe.set_device_channel_indices(np.arange(64)+(64*i))
            probe.set_contact_ids(np.array(probe.to_dataframe()["contact_ids"].values, dtype=np.int64)+(64*i))
            probegroup.add_probe(probe)
        recordingExtractor.set_probegroup(probegroup, group_mode='by_shank', in_place=True)
    #print(recordingExtractor.get_probegroup())

    # plot somewhere to show goemetry of electrodes
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    plot_probe_group(recordingExtractor.get_probegroup())
    #plt.savefig(recording_to_sort+"/"+sorterName+"/"+str(number_of_channels)+"_channel_map.png", dpi=200)
    plt.savefig("/mnt/datastore/Harry/test_recording/probe_locations_"+str(number_of_channels)+"_channels.png", dpi=200)
    plt.close()
    return recordingExtractor


def add_probe_info_by_shank(recordingExtractor, probe_group_df):
    number_of_channels = len(probe_group_df)
    number_of_channels_in_shank = len(probe_group_df[(probe_group_df["probe_index"] == 1) &
                                                     (probe_group_df["shank_ids"] == 1)])
    x = probe_group_df["x"].values[:number_of_channels_in_shank]
    y = probe_group_df["y"].values[:number_of_channels_in_shank]
    geom = np.transpose(np.vstack((x,y)))
    probe = Probe(ndim=2, si_units='um')
    # set circles if tetrodes or rectangle if probe
    if number_of_channels == 16:
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': 5})
    else:
        probe.set_contacts(positions=geom, shapes='rect', shape_params={'width': 11, 'height': 15})
    probe.set_device_channel_indices(np.arange(number_of_channels_in_shank))
    recordingExtractor.set_probe(probe, in_place=True)
    return recordingExtractor


def run_spike_sorting_with_spike_interface(recording_to_sort, sorterName):
    # load signal
    '''
    base_signal = load_OpenEphysRecording(recording_to_sort)
    base_recording = se.NumpyRecording(base_signal,settings.sampling_rate)
    base_recording = add_probe_info(base_recording, recording_to_sort, sorterName)
    base_recording = spre.whiten(base_recording)
    base_recording = spre.bandpass_filter(base_recording, freq_min=300, freq_max=6000)
    bad_channel_ids = getDeadChannel(recording_to_sort +'/dead_channels.txt')
    base_recording.remove_channels(bad_channel_ids)
    base_recording = base_recording.save(folder= settings.temp_storage_path+'/processed', n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)

    params = sorters.get_default_sorter_params(sorterName)
    params['filter'] = False #have already done this in preprocessing step
    params['whiten'] = False
    params['adjacency_radius'] = 200

    sorting = sorters.run_sorter(sorter_name=sorterName, recording=base_recording,output_folder='sorting_tmp', verbose=True, **params)
    sorting = sorting.save(folder= settings.temp_storage_path+'/sorter', n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)
    '''

    params = sorters.get_default_sorter_params(sorterName)
    params['filter'] = False #have already done this in preprocessing step
    params['whiten'] = False
    params['adjacency_radius'] = 200

    # by shark
    n_channels, _ = count_files_that_match_in_folder(recording_to_sort, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')
    probe_group_df = get_probe_dataframe(n_channels)
    bad_channel_ids = getDeadChannel(recording_to_sort +'/dead_channels.txt')

    for probe_index in np.unique(probe_group_df["probe_index"]):
        print("I am subsetting the recording and analysing probe "+str(probe_index))
        probe_df = probe_group_df[probe_group_df["probe_index"] == probe_index]
        for shank_id in np.unique(probe_df["shank_ids"]):
            print("I am looking at shank "+str(shank_id))
            shank_df = probe_df[probe_df["shank_ids"] == shank_id]
            channels_in_shank = np.array(shank_df["channel"])
            base_signal_shank = load_OpenEphysRecording(recording_to_sort, channel_ids=channels_in_shank)
            base_shank_recording = se.NumpyRecording(base_signal_shank,settings.sampling_rate)
            base_shank_recording = add_probe_info_by_shank(base_shank_recording, probe_group_df)
            base_shank_recording = spre.whiten(base_shank_recording)
            base_shank_recording = spre.bandpass_filter(base_shank_recording, freq_min=300, freq_max=6000)
            base_shank_recording.remove_channels(bad_channel_ids)
            base_shank_recording = base_shank_recording.save(folder= settings.temp_storage_path+'/processed_probe'+str(probe_index)+'_shank'+str(shank_id)+'_segment0',
                                                             n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)
            shank_sorting = sorters.run_sorter(sorter_name=sorterName, recording=base_shank_recording, output_folder='sorting_tmp', verbose=True, **params)
            shank_sorting = shank_sorting.save(folder= settings.temp_storage_path+'/sorter_probe'+str(probe_index)+'_shank'+str(shank_id)+'_segment0',
                                               n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)

def get_probe_dataframe(number_of_channels):
    if number_of_channels == 16: # presume tetrodes
        geom = pd.read_csv(settings.tetrode_geom_path, header=None).values
        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': 5})
        probe.set_device_channel_indices(np.arange(number_of_channels))
        probe_df = probe.to_dataframe()
        probe_df["channel"] = np.arange(1,16+1)
        probe_df["shank_ids"] = 1
        probe_df["probe_index"] = 1

    else: # presume cambridge neurotech P1 probes
        assert number_of_channels%64==0
        probegroup = ProbeGroup()
        n_probes = int(number_of_channels/64)
        for i in range(n_probes):
            probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
            probe.move([i*2000, 0]) # move the probes far away from eachother
            probe.set_device_channel_indices(np.arange(64)+(64*i))
            probe.set_contact_ids(np.array(probe.to_dataframe()["contact_ids"].values, dtype=np.int64)+(64*i))
            probegroup.add_probe(probe)
        probe_df = probegroup.to_dataframe()
        probe_df = probe_df.astype({"probe_index": int, "shank_ids": int, "contact_ids": int})
        probe_df["channel"] = np.arange(1,number_of_channels+1)
        probe_df["shank_ids"] = (np.asarray(probe_df["shank_ids"])+1).tolist()
        probe_df["probe_index"] = (np.asarray(probe_df["probe_index"])+1).tolist()
    return probe_df

def sorter2dataframe(sorter, session_id, probe_id, shank_id):
    """Convert data in a sorter to dataframe
        Arguments:
        sorter {SortingExtractor} -- the sorter to extract data frome
        session_id {str} -- ID of the session
    
    Returns:
        pandas.DataFrame -- dataframe containing the spike train, unit featuers and spike features of the sorter
    """

    df = data_frame_utility.df_empty(['session_id', 'probe_id', 'shank_id', 'cluster_id', 'firing_times'],
                                     dtypes=[np.str0, np.uint64, np.uint64, np.uint64, np.uint64])

    # cluster ids identified as probe_id, shank_id, cluster_id 1214 is probe 1, shank 2, cluster 14.
    # This assumes there is less than 10 probes and less than 10 shanks per probe
    for i in sorter.get_unit_ids():
        cell_df = pd.DataFrame()
        cell_df['session_id'] = [session_id]
        cell_df['probe_id'] = [probe_id]
        cell_df['shank_id'] = [shank_id]
        cell_df['cluster_id'] = [int(str(probe_id)+str(shank_id)+str(i))]
        cell_df['firing_times'] = [sorter.get_unit_spike_train(i)]
        df = pd.concat([df, cell_df], ignore_index=True)
    return df
 