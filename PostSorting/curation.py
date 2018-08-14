import os
import json


def load_curation_metrics(spike_data_frame, prm):
    isolations = []
    noise_overlaps = []
    signal_to_noise_ratios = []
    peak_amplitudes = []
    path_to_metrics = prm.get_local_recording_folder_path() + '\\Electrophysiology\\Spike_sorting\\all_tetrodes\\cluster_metrics.json'
    if os.path.exists(path_to_metrics):
        with open(path_to_metrics) as metrics_file:
            cluster_metrics = json.load(metrics_file)
            metrics_file.close()
        for cluster in range(len(spike_data_frame)):
            isolation = cluster_metrics["clusters"][cluster]["metrics"]["isolation"]
            noise_overlap = cluster_metrics["clusters"][cluster]["metrics"]["noise_overlap"]
            peak_snr = cluster_metrics["clusters"][cluster]["metrics"]["peak_snr"]
            peak_amp = cluster_metrics["clusters"][cluster]["metrics"]["peak_amp"]

            isolations.append(isolation)
            noise_overlaps.append(noise_overlap)
            signal_to_noise_ratios.append(peak_snr)
            peak_amplitudes.append(peak_amp)

        spike_data_frame['isolation'] = isolations
        spike_data_frame['noise_overlap'] = noise_overlaps
        spike_data_frame['peak_snr'] = signal_to_noise_ratios
        spike_data_frame['peak_amp'] = peak_amplitudes
    return spike_data_frame

