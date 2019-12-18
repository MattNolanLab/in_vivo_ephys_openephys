import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""
    Author: Harry Clark 1/7/19
    This script contains functions for matching waveforms by pca

"""

def add_snippets_pc(spatial_firing, snippet_dataframe=None):
    # snippet dataframe should be created in the same location as spatial firing and used
    # to add principal components to spatial firing from all snippets

    clusters_pcs = []
    for cluster_id in spatial_firing["cluster_id"]:
        #snippets = np.asarray(spatial_firing1.loc[spatial_firing1.loc[:, "cluster_id"] == cluster_id, :]["all_snippets"])[0]
        snippets = np.asarray(spatial_firing.loc[spatial_firing.loc[:, "cluster_id"] == cluster_id, :]["random_snippets"])[0]

        snippets_pc_channels = []
        for i in range(len(snippets)): # number of tetrodes
            channel_snippets = StandardScaler().fit_transform(snippets[i].transpose())
            pca = PCA(n_components=2)
            snippets_pc = pca.fit_transform(channel_snippets)

            # add snippets across each channel
            snippets_pc_channels.append(snippets_pc)

        # add snippets across all clusters
        clusters_pcs.append(snippets_pc_channels)

    # append spatial firing dataframe
    spatial_firing["snippets_pc"] = clusters_pcs

    return spatial_firing


def waveform_matching(spatial_firing1, spatial_firing2):
    '''
    Manual matching takes two spatial_firing dataframes and looks to resolve the same identified cluster across two recordings
    :param spatial_firing1:
    :param spatial_firing2:
    :returns spatial_firing1, spatial_firing2 with appended field for matched cluster ids (null if not matched)
    '''

    # snippet data is size [1][4][30][50]

    for cluster_id1 in spatial_firing1["cluster_id"]:
        for cluster_id2 in spatial_firing2["cluster_id"]:

            #fig = plt.figure(figsize=(8, 8))
            #ax = fig.add_subplot(1, 1, 1)
            #ax.set_xlabel('Principal Component 1', fontsize=15)
            #ax.set_ylabel('Principal Component 2', fontsize=15)
            #ax.set_title('2 component PCA', fontsize=20)

            snippets_pc1 = np.asarray(spatial_firing1.loc[spatial_firing1.loc[:, "cluster_id"] == cluster_id1, :]["snippets_pc"])[0]
            snippets_pc2 = np.asarray(spatial_firing2.loc[spatial_firing2.loc[:, "cluster_id"] == cluster_id2, :]["snippets_pc"])[0]

            #get tetrode number, if tetrode number is same, then make comparison and plot



        '''
        snippets_pc_tetrode = []
        for i in range(len(snippets_1)):
            channel_snippets = StandardScaler().fit_transform(snippets_1[i].transpose())
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(channel_snippets)

            snippets_pc_tetrode.append(principalComponents)
            #ax.scatter(principalComponents[:,0], principalComponents[:,1], label=str(cluster_id))
        ax.legend()
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        '''

    # instantiate new collumn and assign
    #spatial_firing1["matched_cluster_ids"] = matched_clusters1
    #spatial_firing1["matched_cluster_ids"] = matched_clusters2

    return spatial_firing1, spatial_firing2


def test_waveform_matching(vr_spatial_firing, of_spatial_firing):
    waveform_matching(vr_spatial_firing, of_spatial_firing)


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    '''
    vr_spatial_firing = pd.read_pickle(
        '/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/MouseVR/Sarah_cohort4/M2_D9_2019-06-27_13-58-07/MountainSort/DataFrames/spatial_firing.pkl')
    of_spatial_firing = pd.read_pickle(
        '/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Harry/MouseOF/M2_D9_2019-06-27_14-40-06/MountainSort/DataFrames/spatial_firing.pkl')

    #test_waveform_matching(vr_spatial_firing, of_spatial_firing)

    # this holds a good spatial firing from m2 from cohort 4 with some ramping cells, this is the vr recording
    vr_spatial_firing = pd.read_pickle('/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Sarah/Data/PIProject_OptoEphys/Data/OpenEphys/_cohort4/VirtualReality/M2_D3_2019-03-06_13-35-15/MountainSort/DataFrames/spatial_firing.pkl')
    # i need to run the open field recording to get the spatial firing
    of_spatial_firing = pd.read_pickle('/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Sarah/Data/test/openfield/M2_D3_2019-03-06_15-24-38/MountainSort/DataFrames/spatial_firing.pkl')
    '''

    #vr_spatial_firing = pd.read_pickle('/home/harry/Desktop/test_wf_matching/vr/spatial_firing.pkl')
    #of_spatial_firing = pd.read_pickle('/home/harry/Desktop/test_wf_matching/of/spatial_firing.pkl')

    of_spatial_firing = pd.read_pickle(r"C:\Users\44756\Google Drive\PhD Edinburgh\in vivo\test_wf_matching\of\spatial_firing.pkl")
    vr_spatial_firing = pd.read_pickle(r"C:\Users\44756\Google Drive\PhD Edinburgh\in vivo\test_wf_matching\vr\spatial_firing.pkl")

    vr_spatial_firing = add_snippets_pc(vr_spatial_firing)
    of_spatial_firing = add_snippets_pc(of_spatial_firing)

    test_waveform_matching(vr_spatial_firing, of_spatial_firing)

if __name__ == '__main__':
    main()