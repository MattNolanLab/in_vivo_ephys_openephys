import ManualCuration.manual_curation_settings
import numpy as np
import pandas as pd


def load_phy_output(recording_local):
    print('Loading phy output...')
    path_to_phy_output = recording_local + '/MountainSort/phy/'
    spike_times = np.load(path_to_phy_output + 'spike_times.npy')
    spike_clusters = np.load(path_to_phy_output + 'spike_clusters.npy')
    cluster_group = pd.read_csv(path_to_phy_output + 'cluster_group.tsv', sep="\t")
    print('Clusters loaded:')
    print(cluster_group)
    if cluster_group.group.str.contains('unsorted').sum() > 0:
        print('WARNING: There are unsorted clusters in this file. '
              'You should mark each cluster as good, noise or MUA during curation. You should be running this script '
              'after doing the manual curation in phy. '
              'https://phy.readthedocs.io/en/latest/clustering/')
    return spike_times, spike_clusters, cluster_group


def post_process_manually_curated_data(recording_server, recording_local):
    spike_times, spike_clusters, cluster_group = load_phy_output(recording_local)
    # read phy output and split firing times back and save as spatial_firing_curated (also for paired recordings)
    # save output back on server (copy manual spatial firing back)
    ##do after this: change pipeline so it loads manual spatial firing if it exists (?)


def main():
    recording_server = ManualCuration.manual_curation_settings.get_recording_path_datastore()
    recording_local = ManualCuration.manual_curation_settings.get_local_recording_path()
    print('This script will process the manually curated phy outputs and upload them to datastore: ' + recording_server)
    post_process_manually_curated_data(recording_server, recording_local)


if __name__ == '__main__':
    main()