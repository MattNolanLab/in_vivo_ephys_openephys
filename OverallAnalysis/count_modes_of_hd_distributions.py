import datetime
import gc
import numpy as np
import pandas as pd
from rpy2 import robjects as robj
from rpy2.robjects import pandas2ri

local_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df_all2.pkl'
output_path = '/Users/s1466507/Documents/Ephys/recordings/all_mice_df_all_hd_modes.pkl'
script_path = '/Users/s1466507/Documents/GitHub/in_vivo_ephys_openephys/OverallAnalysis/'


def format_input_for_r(cluster):
    cluster_hd_np = np.array(cluster.hd)
    cluster_hd_np = cluster_hd_np[~np.isnan(cluster_hd_np)]
    cluster_hd_np = np.round(cluster_hd_np, 4)
    cluster_hd_np = cluster_hd_np * np.pi / 180
    hd_cluster = pd.DataFrame({'hd': cluster_hd_np})
    hd_cluster_r = pandas2ri.py2ri(hd_cluster)
    return hd_cluster_r


def count_modes_of_hd_distributions():
    max_number_of_modes_to_try_to_fit = 30
    spatial_firing = pd.read_pickle(local_path)
    print(spatial_firing.head())
    number_of_modes_all_clusters = []
    aic_number_of_modes = []
    robj.r.source('count_modes_circular.R')

    for index, cluster in spatial_firing.iterrows():
        try:
            print(cluster.session_id)
            print(index)
            if cluster.watson_test_hd < 0.385 or cluster.mean_firing_rate > 10:
                number_of_modes_all_clusters.append(np.nan)
                continue
            hd_cluster_r = format_input_for_r(cluster)
            convert_to_cart_coord_r = robj.r['cart_coord']
            cart_coord_hd = convert_to_cart_coord_r(hd_cluster_r)
            gc.collect()

            print(datetime.datetime.now())

            fit_mixed_models = robj.r['fit_mixed_models']
            fit = fit_mixed_models(cart_coord_hd, max_number_of_modes_to_try_to_fit)
            gc.collect()
            get_aic_r = robj.r['get_aic']
            aic = get_aic_r(fit, max_number_of_modes_to_try_to_fit)
            aic_number_of_modes.append(aic)
            print(aic)
            number_of_modes = np.argmin(aic) + 1
            number_of_modes_all_clusters.append(number_of_modes)
            print('Number of modes is: ' + str(number_of_modes))

        except:
            print('There was a problem with processing this cluster.')
            print(number_of_modes_all_clusters)
            # number_of_modes_all_clusters.append(np.nan)
            # might want to append array with nan

    spatial_firing['number_of_modes_hd'] = number_of_modes_all_clusters
    spatial_firing['number_of_modes_hd_aic'] = aic_number_of_modes
    print(spatial_firing.columns)
    print(spatial_firing.head())
    return spatial_firing


def main():
    spatial_firing = count_modes_of_hd_distributions()
    spatial_firing.to_pickle(local_path)


if __name__ == '__main__':
    main()
