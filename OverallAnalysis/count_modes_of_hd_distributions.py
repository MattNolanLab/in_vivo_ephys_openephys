import numpy as np
import os
import pandas as pd
from rpy2 import robjects as robj
from rpy2.robjects import pandas2ri

local_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df_all.pkl'


def count_modes_of_hd_distributions():
    spatial_firing = pd.read_pickle(local_path)
    print(spatial_firing.head())

    for index, cluster in spatial_firing.iterrows():
        # todo need to convert this to an R object http://rpy.sourceforge.net/rpy2/doc-2.2/html/vector.html
        # hd_cluster_r = r.vectors.FloatVector(hd_cluster)
        cluster_hd_np = np.array(cluster.hd)
        cluster_hd_np = cluster_hd_np[~np.isnan(cluster_hd_np)]
        cluster_hd_np = np.round(cluster_hd_np, 4)
        hd_cluster = pd.DataFrame({'hd':cluster_hd_np})
        hd_cluster_r = pandas2ri.py2ri(hd_cluster)
        if index < 20:
            continue
        # hd_cluster = list(cluster.hd)
        # np.savetxt('hd_cluster.csv', hd_cluster)
        # hd_cluster = robj.DataFrame({'label':range(len(hd_cluster)),'hd':cluster.hd})

        # path = os.path.dirname(os.path.realpath(__file__)) + 'hd_cluster_csv'
        # path_r = r.vectors.StrVector(path)
        # r_vector = robj.FloatVector(hd_cluster)
        robj.r.source('count_modes_circular.R')
        #print(hd_cluster)
        # aic = r.r['count_number_of_modes'](r_vector)  # turn this back to python object
        rcall = robj.r['count_modes']
        test = rcall(hd_cluster_r)
        print(test)

    # rerun load df function to get hd from all recordings...

    # loop through clusters

        # get hd from cluster
        # call R sctipt somehow
        # R script
        # get output of R script
        # put result back in df (number of modes


def main():
    count_modes_of_hd_distributions()


if __name__ == '__main__':
    main()
