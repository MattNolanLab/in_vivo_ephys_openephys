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

def load_OpenEphysRecording(folder):
    number_of_channels, corrected_data_file_suffix = count_files_that_match_in_folder(folder, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')
    signal = []
    for i in range(number_of_channels):
        fname = folder+'/'+corrected_data_file_suffix+str(i+1)+settings.data_file_suffix+'.continuous'
        x = OpenEphys.loadContinuousFast(fname)['data']
        if i==0:
            #preallocate array on first run
            signal = np.zeros((x.shape[0], number_of_channels))
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
    print(recordingExtractor.get_probegroup())

    # plot somewhere to show goemetry of electrodes
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    plot_probe_group(recordingExtractor.get_probegroup())
    #plt.savefig(recording_to_sort+"/"+sorterName+"/"+str(number_of_channels)+"_channel_map.png", dpi=200)
    plt.savefig("/mnt/datastore/Harry/test_recording/probe_locations_"+str(number_of_channels)+"_channels.png", dpi=200)
    plt.close()
    return recordingExtractor


def run_spike_sorting_with_spike_interface(recording_to_sort, sorterName):
    # load signal
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
    return sorting

def sorter2dataframe(sorter, session_id):
    """Convert data in a sorter to dataframe
        Arguments:
        sorter {SortingExtractor} -- the sorter to extract data frome
        session_id {str} -- ID of the session
    
    Returns:
        pandas.DataFrame -- dataframe containing the spike train, unit featuers and spike features of the sorter
    """

    df = data_frame_utility.df_empty(['session_id', 'cluster_id', 'primary_channel', 'firing_times'],
                                     dtypes=[np.str0, np.uint8, np.uint8, np.uint64])

    for i in sorter.get_unit_ids():
        cell_df = pd.DataFrame()
        cell_df['session_id'] = [session_id]
        cell_df['cluster_id'] = [i]
        cell_df['firing_times'] = [sorter.get_unit_spike_train(i)]
        df = pd.concat([df, cell_df], ignore_index=True)
    return df
 