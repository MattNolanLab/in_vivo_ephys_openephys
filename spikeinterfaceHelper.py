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

    channel_ids = np.sort(channel_ids)
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

def get_wiring_for_cambridgeneurotech_ASSY_236_P_1(contact_id, probe_id):
    contact_id = contact_id-(64*probe_id)

    if contact_id == 1:
        return 41 + (64*probe_id)
    elif contact_id == 2:
        return 39 + (64*probe_id)
    elif contact_id == 3:
        return 38 + (64*probe_id)
    elif contact_id == 4:
        return 37 + (64*probe_id)
    elif contact_id == 5:
        return 35 + (64*probe_id)
    elif contact_id == 6:
        return 34 + (64*probe_id)
    elif contact_id == 7:
        return 33 + (64*probe_id)
    elif contact_id == 8:
        return 32 + (64*probe_id)
    elif contact_id == 9:
        return 29 + (64*probe_id)
    elif contact_id == 10:
        return 30 + (64*probe_id)
    elif contact_id == 11:
        return 28 + (64*probe_id)
    elif contact_id == 12:
        return 26 + (64*probe_id)
    elif contact_id == 13:
        return 25 + (64*probe_id)
    elif contact_id == 14:
        return 24 + (64*probe_id)
    elif contact_id == 15:
        return 22 + (64*probe_id)
    elif contact_id == 16:
        return 20 + (64*probe_id)
    elif contact_id == 17:
        return 46 + (64*probe_id)
    elif contact_id == 18:
        return 45 + (64*probe_id)
    elif contact_id == 19:
        return 44 + (64*probe_id)
    elif contact_id == 20:
        return 43 + (64*probe_id)
    elif contact_id == 21:
        return 42 + (64*probe_id)
    elif contact_id == 22:
        return 40 + (64*probe_id)
    elif contact_id == 23:
        return 36 + (64*probe_id)
    elif contact_id == 24:
        return 31 + (64*probe_id)
    elif contact_id == 25:
        return 27 + (64*probe_id)
    elif contact_id == 26:
        return 23 + (64*probe_id)
    elif contact_id == 27:
        return 21 + (64*probe_id)
    elif contact_id == 28:
        return 18 + (64*probe_id)
    elif contact_id == 29:
        return 19 + (64*probe_id)
    elif contact_id == 30:
        return 17 + (64*probe_id)
    elif contact_id == 31:
        return 16 + (64*probe_id)
    elif contact_id == 32:
        return 14 + (64*probe_id)
    elif contact_id == 33:
        return 55 + (64*probe_id)
    elif contact_id == 34:
        return 53 + (64*probe_id)
    elif contact_id == 35:
        return 54 + (64*probe_id)
    elif contact_id == 36:
        return 52 + (64*probe_id)
    elif contact_id == 37:
        return 51 + (64*probe_id)
    elif contact_id == 38:
        return 50 + (64*probe_id)
    elif contact_id == 39:
        return 49 + (64*probe_id)
    elif contact_id == 40:
        return 48 + (64*probe_id)
    elif contact_id == 41:
        return 47 + (64*probe_id)
    elif contact_id == 42:
        return 15 + (64*probe_id)
    elif contact_id == 43:
        return 13 + (64*probe_id)
    elif contact_id == 44:
        return 12 + (64*probe_id)
    elif contact_id == 45:
        return 11 + (64*probe_id)
    elif contact_id == 46:
        return 9 + (64*probe_id)
    elif contact_id == 47:
        return 10 + (64*probe_id)
    elif contact_id == 48:
        return 8 + (64*probe_id)
    elif contact_id == 49:
        return 63 + (64*probe_id)
    elif contact_id == 50:
        return 62 + (64*probe_id)
    elif contact_id == 51:
        return 61 + (64*probe_id)
    elif contact_id == 52:
        return 60 + (64*probe_id)
    elif contact_id == 53:
        return 59 + (64*probe_id)
    elif contact_id == 54:
        return 58 + (64*probe_id)
    elif contact_id == 55:
        return 57 + (64*probe_id)
    elif contact_id == 56:
        return 56 + (64*probe_id)
    elif contact_id == 57:
        return 7 + (64*probe_id)
    elif contact_id == 58:
        return 6 + (64*probe_id)
    elif contact_id == 59:
        return 5 + (64*probe_id)
    elif contact_id == 60:
        return 4 + (64*probe_id)
    elif contact_id == 61:
        return 3 + (64*probe_id)
    elif contact_id == 62:
        return 2 + (64*probe_id)
    elif contact_id == 63:
        return 1 + (64*probe_id)
    elif contact_id == 64:
        return 0 + (64*probe_id)
    else:
        print("contact is invalid")



def get_wiring(contact_ids, probes_ids, probe_manufacturer, probe_type):
    # add wiring info here when new experiments use different probes
    wiring_ids = []
    if probe_manufacturer == "cambridgeneurotech" and probe_type == "ASSY-236-P-1":
        for contact_id, probe_id in zip(contact_ids, probes_ids):
            corresponding_wiring_id = get_wiring_for_cambridgeneurotech_ASSY_236_P_1(contact_id, probe_id)
            wiring_ids.append(corresponding_wiring_id)
    else:
        print("The given probe_manufacturer and probe_type do not have the wiring set yet"
              "Check the arguments and add the wiring here if not yet added")
    return np.array(wiring_ids)

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
            contact_ids = np.array(probe.to_dataframe()["contact_ids"].values, dtype=np.int64)+(64*i)
            probe.set_contact_ids(contact_ids)
            probe.set_device_channel_indices(get_wiring(contact_ids, 'cambridgeneurotech', 'ASSY-236-P-1'))
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


def add_probe_info_by_shank(recordingExtractor, shank_df):
    shank_df = shank_df.sort_values(by="channel", ascending=True)
    x = shank_df["x"].values
    y = shank_df["y"].values
    geom = np.transpose(np.vstack((x,y)))
    probe = Probe(ndim=2, si_units='um')
    if shank_df["contact_shapes"].iloc[0] == "rect":
        probe.set_contacts(positions=geom, shapes='rect', shape_params={'width': shank_df["width"].iloc[0],
                                                                        'height': shank_df["height"].iloc[0]})
    elif shank_df["contact_shapes"].iloc[0] == "circle":
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': shank_df["radius"].iloc[0]})
    probe.set_device_channel_indices(np.arange(len(shank_df)))
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
    params['num_workers'] = 2

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
            base_shank_recording = add_probe_info_by_shank(base_shank_recording, shank_df)
            base_shank_recording = spre.whiten(base_shank_recording)
            base_shank_recording = spre.bandpass_filter(base_shank_recording, freq_min=300, freq_max=6000)
            base_shank_recording.remove_channels(bad_channel_ids)
            base_shank_recording = base_shank_recording.save(folder= settings.temp_storage_path+'/processed_probe'+str(probe_index)+'_shank'+str(shank_id)+'_segment0',
                                                             n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)
            shank_sorting = sorters.run_sorter(sorter_name=sorterName, recording=base_shank_recording, output_folder='sorting_tmp', verbose=True, **params)
            shank_sorting = shank_sorting.save(folder= settings.temp_storage_path+'/sorter_probe'+str(probe_index)+'_shank'+str(shank_id)+'_segment0',
                                               n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)
    return

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
            contact_ids = np.array(probe.to_dataframe()["contact_ids"].values, dtype=np.int64)+(64*i)
            #probe.set_device_channel_indices(get_wiring(contact_ids, 'cambridgeneurotech', 'ASSY-236-P-1', probe_i=i))
            probe.set_contact_ids(contact_ids)
            probegroup.add_probe(probe)
        probe_df = probegroup.to_dataframe()
        probe_df = probe_df.astype({"probe_index": int, "shank_ids": int, "contact_ids": int})
        probe_df["channel"] = get_wiring(probe_df["contact_ids"], probe_df["probe_index"], 'cambridgeneurotech', 'ASSY-236-P-1')+1
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
                                     dtypes=[np.str0, np.str0, np.str0, np.str0, np.uint64])

    # cluster ids identified as probe_id, shank_id, cluster_id 1214 is probe 1, shank 2, cluster 14.
    # This assumes there is less than 10 probes and less than 10 shanks per probe
    for i in sorter.get_unit_ids():
        cell_df = pd.DataFrame()
        cell_df['session_id'] = [session_id]
        cell_df['probe_id'] = [probe_id]
        cell_df['shank_id'] = [shank_id]
        cell_df['cluster_id'] = [str(int(str(probe_id)+str(shank_id)+str(i)))]
        cell_df['firing_times'] = [sorter.get_unit_spike_train(i)]
        df = pd.concat([df, cell_df], ignore_index=True)
    return df
 