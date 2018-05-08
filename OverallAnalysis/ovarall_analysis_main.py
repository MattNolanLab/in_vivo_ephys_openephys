#  This script is to perform overall analysis on multiple days recorded from a group of animals
import OverallAnalysis.organize_cluster_data

path_to_data = 'C:/Users/s1466507/Desktop'


def construct_data_frame():
    OverallAnalysis.organize_cluster_data.get_sorting_output()
    pass


def run_analyses():
    pass


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    construct_data_frame()
    run_analyses()


if __name__ == '__main__':
    main()