import control_sorting_analysis
import os
import json
import pandas as pd
import spikeinterface as si
import spikeinterface.qualitymetrics as qm
import settings
import numpy as np
from spikeinterface.postprocessing import compute_principal_components

ignore_curation = False


def load_curation_metrics(spike_data_frame, sorter_name, local_recording_folder_path, ms_tmp_path):
    isolations = []
    noise_overlaps = []
    signal_to_noise_ratios = []
    peak_amplitudes = []
    sorter_name = sorter_name
    path_to_metrics = local_recording_folder_path + '/Electrophysiology/' + sorter_name + '/cluster_metrics.json'
    if not os.path.exists(path_to_metrics):
        print('I did not find the curation results.')

        for filename in os.listdir(ms_tmp_path + 'prvbucket/_mountainprocess/'):
            if filename.startswith('output_metrics_out'):
                print(filename)
                path_to_metrics = ms_tmp_path + '/prvbucket/_mountainprocess/' + filename
                
    if os.path.exists(path_to_metrics):
        with open(path_to_metrics) as metrics_file:
            cluster_metrics = json.load(metrics_file)
            metrics_file.close()
        for cluster_index, cluster in spike_data_frame.iterrows():
            for cluster_metric_index in range(len(cluster_metrics["clusters"])):
                if cluster_metrics["clusters"][cluster_metric_index]["label"] == cluster.cluster_id:
                    isolation = cluster_metrics["clusters"][cluster_metric_index]["metrics"]["isolation"]
                    noise_overlap = cluster_metrics["clusters"][cluster_metric_index]["metrics"]["noise_overlap"]
                    peak_snr = cluster_metrics["clusters"][cluster_metric_index]["metrics"]["peak_snr"]
                    peak_amp = cluster_metrics["clusters"][cluster_metric_index]["metrics"]["peak_amp"]

                    isolations.append(isolation)
                    noise_overlaps.append(noise_overlap)
                    signal_to_noise_ratios.append(peak_snr)
                    peak_amplitudes.append(peak_amp)

        spike_data_frame['isolation'] = isolations
        spike_data_frame['noise_overlap'] = noise_overlaps
        spike_data_frame['peak_snr'] = signal_to_noise_ratios
        spike_data_frame['peak_amp'] = peak_amplitudes
    return spike_data_frame

def save_waveforms_locally(we, save_folder_path, cluster_ids):
    if os.path.exists(save_folder_path) is False:
        os.makedirs(save_folder_path)
    for id in cluster_ids:
        waveforms = we.get_waveforms(unit_id=id)
        np.save(save_folder_path+"waveforms_"+str(id)+".npy", np.array(waveforms))

def add_primary_channels(spike_data_frame, we):
    channel_ids = si.get_template_extremum_channel(we)
    primary_channels = []
    for cluster_index, cluster in spike_data_frame.iterrows():
        cluster_id = cluster["cluster_id"]
        primary_channel = channel_ids[cluster_id]
        primary_channels.append(primary_channel+1)
    spike_data_frame["primary_channel"] = primary_channels
    return spike_data_frame

def curate_data(spike_data_frame, sorter_name, local_recording_folder_path, ms_tmp_path, SorterInstance=None):
    # first check for manual curation
    manually_curated = control_sorting_analysis.check_if_curated_data_is_available(local_recording_folder_path)
    if manually_curated:
        return spike_data_frame, pd.DataFrame()
    elif 'isolation' in spike_data_frame:
        noisy_cluster = pd.DataFrame()
        noisy_cluster['this is empty'] = 'Noisy clusters were not reloaded. Sort again if you need them.'
        return spike_data_frame, noisy_cluster
    elif SorterInstance is not None: # add curation metrics from spike interface

        cluster_ids = spike_data_frame["cluster_id"].values.tolist()
        # load the sorting and recording extractor from the tmp folder as curation is done across all valid recordings
        RecordingInstance = si.load_extractor(settings.temp_storage_path+'/processed')
        SorterInstance = si.load_extractor(settings.temp_storage_path+'/sorter')
        we = si.extract_waveforms(RecordingInstance, SorterInstance, folder=settings.temp_storage_path+"/waveforms", ms_before=1, ms_after=1, load_if_exists=False, overwrite=True)
        save_waveforms_locally(we, settings.temp_storage_path+'/waveform_arrays/', cluster_ids)
        pca = compute_principal_components(we, n_components=5, mode='by_channel_local')
        quality_metrics = qm.compute_quality_metrics(we, n_jobs = 4, metric_names=['snr','isi_violation','firing_rate', 'presence_ratio', 'amplitude_cutoff',
                                                                                   'isolation_distance', 'l_ratio', 'd_prime', 'nearest_neighbor', 'nn_isolation', 'nn_noise_overlap'])
        quality_metrics["cluster_id"] = cluster_ids
        spike_data_frame = spike_data_frame.merge(quality_metrics, on='cluster_id')
        spike_data_frame = add_primary_channels(spike_data_frame, we)
        isolation_threshold = 0.9
        noise_overlap_threshold = 0.05
        snr_threshold = 1
        firing_rate_threshold = 0

        isolated_cluster = spike_data_frame['nn_isolation'] > isolation_threshold
        low_noise_cluster = spike_data_frame['nn_noise_overlap'] < noise_overlap_threshold
        high_snr = spike_data_frame['snr'] > snr_threshold
        high_mean_firing_rate = spike_data_frame['mean_firing_rate'] > firing_rate_threshold

        good_cluster = spike_data_frame[isolated_cluster & low_noise_cluster & high_snr & high_mean_firing_rate].copy()
        noisy_cluster = spike_data_frame.loc[~spike_data_frame.index.isin(list(good_cluster.index))]

    else: # otherwise load the curation metrics from mountainsort3
        spike_data_frame = load_curation_metrics(spike_data_frame, sorter_name, local_recording_folder_path, ms_tmp_path)
        isolation_threshold = 0.9
        noise_overlap_threshold = 0.05
        peak_snr_threshold = 1
        firing_rate_threshold = 0

        isolated_cluster = spike_data_frame['isolation'] > isolation_threshold
        low_noise_cluster = spike_data_frame['noise_overlap'] < noise_overlap_threshold
        high_peak_snr = spike_data_frame['peak_snr'] > peak_snr_threshold
        high_mean_firing_rate = spike_data_frame['mean_firing_rate'] > firing_rate_threshold

        good_cluster = spike_data_frame[isolated_cluster & low_noise_cluster & high_peak_snr & high_mean_firing_rate].copy()
        noisy_cluster = spike_data_frame.loc[~spike_data_frame.index.isin(list(good_cluster.index))]

    if ignore_curation:
        good_cluster['Curated']=True
        noisy_cluster['Curated']=False
        return pd.concat([good_cluster, noisy_cluster]), pd.DataFrame()

    return good_cluster, noisy_cluster


