import pandas as pd
from rpy2 import robjects as r

local_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df_all.pkl'


def count_modes_of_hd_distributions():
    spatial_firing = pd.read_pickle(local_path)
    print(spatial_firing.head())

    for index, cluster in spatial_firing.iterrows():
        hd_cluster = cluster.hd  # todo need to convert this to an R object http://rpy.sourceforge.net/rpy2/doc-2.2/html/vector.html
        r.r.source('count_modes_of_circular_dist.R')
        aic = r.r['count_number_of_modes'](hd_cluster)  # turn this back to python object

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
