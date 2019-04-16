import pandas as pd


local_path = '/Users/s1466507/Dropbox/Edinburgh/PhD/thesis/4 opto_tagging_open_field/plot_evoked_vs_normal/'


def process_data():
    spatial_firing_path = local_path + 'spatial_firing.pkl'
    spatial_firing = pd.read_pickle(spatial_firing_path)
    print(spatial_firing.head)


def main():
    process_data()


if __name__ == '__main__':
    main()