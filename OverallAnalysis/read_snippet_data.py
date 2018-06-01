import tables
import pandas as pd

def get_snippets(filename, path_to_data):
    path = path_to_data + filename + '/' + 'Firings0.mat'
    firings = tables.openFile(path)
    cluster_id = firings.root.cluid[:]
    cluster_id.flatten()
    spike_index = firings.root.spikeind[:]
    spike_index.flatten()
    waveforms = firings.root.waveforms[:]
    return cluster_id, spike_index, waveforms


def analyze_snippets(dataframe, path_to_data):
    cluster_id, spike_index, waveforms = get_snippets('M5_2018-03-06_15-34-44_of', path_to_data)

