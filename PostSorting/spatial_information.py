'''
Function for calculating the spatial information of cells

'''

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def get_spatial_info(rate_map,occ, min_dwell=3):
    # return spatial information in terms of bit per spike
    # rate_map: firing rate
    # occ: occupancy in each bin (raw histogram, unnormalized)
    # min_well: minimum datapoint for which a bin is considered
    # as in W. E. Skaggs, B. L. McNaughton, and K. M. Gothard, “An Information-Theoretic Approach to Deciphering the Hippocampal Code,” 



    mean_fr = rate_map.mean()
    
    #make it a linear matrix for easier computation
    occ = occ.ravel()
    occ_map = occ / occ.sum()
    fr = rate_map.ravel()
    
    H1 = 0
    for i in range(len(fr)):
        if fr[i]>0 and occ[i]>min_dwell:
            H1 += fr[i] * np.log2(fr[i]/mean_fr)*occ_map[i]
    
    return H1/mean_fr


def process_spatial_info(spatial_firing, pos_df):
    # Compute spatial information for each luster
    # spatial_firing: dataframe containing the position for each spike
    # position: dataframe containing the whole position info of the animal in openfield

    position_x = pos_df.position_x
    position_y = pos_df.position_y

    position = np.stack([position_x,position_y]).T
    pos_scaler = MinMaxScaler((0,100)).fit(position) # use scaler so that we can scale the spatial_firing dat atoo
    position_norm = pos_scaler.transform(position)

    # Get the occpancy in each bin
    bins = np.arange(0,100,2.5)
    occ, xedge, yedge = np.histogram2d(position_norm[:,0], position_norm[:,1],bins) #40 bins, i.e. 2.5cm per bin

    spatial_info = []

    for i in range(len(spatial_firing)):
        # get the firing rate map
        spike_pos_x = spatial_firing.iloc[i].position_x
        spike_pos_y = spatial_firing.iloc[i].position_y

        spike_loc = np.stack([spike_pos_x, spike_pos_y]).T
        spike_loc_norm = pos_scaler.transform(spike_loc)

        spike_count,_,_ = np.histogram2d(spike_loc_norm[:,0], spike_loc_norm[:,1],bins) #40 bins, i.e. 2.5cm per bin


        with np.errstate(divide='ignore',invalid='ignore'):
            rate_map = spike_count / occ * 30 # 30Hz camera FPS
        rate_map[np.isnan(rate_map)] = 0 #remove bin with low occupancy 

        spatial_info.append(get_spatial_info(rate_map,occ))

    spatial_info_df = spatial_firing.loc[:,['session_id','cluster_id']].copy()
    spatial_info_df['spatial_info'] = spatial_info

    return spatial_info_df