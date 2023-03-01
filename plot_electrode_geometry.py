import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.extractors as se
from probeinterface import get_probe
from probeinterface.plotting import plot_probe
from probeinterface import Probe, ProbeGroup
from probeinterface.plotting import plot_probe_group

def test_probe_interface(save_path):
    recording, sorting_true = se.toy_example(duration=1, num_channels=64, seed=0, num_segments=4)
    probe1 = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
    probe1.set_device_channel_indices(np.arange(64))
    print(probe1)
    recording_probes = recording.set_probe(probe1, group_mode='by_shank')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    plot_probe(recording_probes.get_probe(), ax=ax, with_channel_index=True)
    plt.savefig("/mnt/datastore/Harry/test_recording/probe_locations_"+str(64)+"_channels.png", dpi=200)
    plt.close()

def test_double_probe_interface(save_path):
    recording, sorting_true = se.toy_example(duration=1, num_channels=128, seed=0, num_segments=4)
    probe1 = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
    probe2 = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
    probe2.move([2000, 0])
    probe1.set_device_channel_indices(np.arange(64))
    probe2.set_device_channel_indices(np.arange(64, 128))
    probe2.set_contact_ids(np.array(probe1.to_dataframe()["contact_ids"].values, dtype=np.int64)+64)
    probegroup = ProbeGroup()
    probegroup.add_probe(probe1)
    probegroup.add_probe(probe2)
    print(probegroup)
    recording_probes = recording.set_probegroup(probegroup, group_mode='by_shank')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    plot_probe_group(recording_probes.get_probegroup(), ax=ax)
    plt.savefig("/mnt/datastore/Harry/test_recording/probe_locations_"+str(128)+"_channels.png", dpi=200)
    plt.close()

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    save_path = "/mnt/datastore/Harry/test_recording/"
    #test_probe_interface(save_path)
    test_double_probe_interface(save_path)
    print("look now")

if __name__ == '__main__':
    main()
