import numpy as np
import os
import pandas as pd
from rpy2 import robjects as robj
from rpy2.robjects import pandas2ri

local_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df_all.pkl'
output_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df_all_hd_modes.pkl'


def count_modes_of_hd_distributions():
    spatial_firing = pd.read_pickle(local_path)
    print(spatial_firing.head())
    number_of_modes_all_clusters = []
    aic_number_of_modes = []
    for index, cluster in spatial_firing.iterrows():
        cluster_hd_np = np.array(cluster.hd)
        cluster_hd_np = cluster_hd_np[~np.isnan(cluster_hd_np)]
        cluster_hd_np = np.round(cluster_hd_np, 4)
        hd_cluster = pd.DataFrame({'hd':cluster_hd_np})
        hd_cluster_r = pandas2ri.py2ri(hd_cluster)
        robj.r.source('count_modes_circular.R')
        rcall = robj.r['count_modes']
        aic = rcall(hd_cluster_r)
        aic_number_of_modes.append(aic)
        print(aic)
        number_of_modes = np.argmin(aic) + 1
        number_of_modes_all_clusters.append(number_of_modes)
    spatial_firing['number_od_modes_hd'] = number_of_modes_all_clusters
    spatial_firing['number_od_modes_hd_aic'] = aic_number_of_modes
    print(spatial_firing.columns)
    print(spatial_firing.head())
    return spatial_firing


def main():
    spatial_firing = count_modes_of_hd_distributions()
    spatial_firing.to_pickle(local_path)


if __name__ == '__main__':
    main()
