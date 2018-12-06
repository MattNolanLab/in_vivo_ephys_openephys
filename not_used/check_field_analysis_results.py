import os
import glob


server_path = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Klara/Open_field_opto_tagging_p038/'


def process_recordings():
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        firing_fields_path = recording_folder + '/MountainSort/Figures/field_analysis'
        if os.path.isdir(firing_fields_path):
            if len(os.listdir(firing_fields_path)) != 0:
                print(firing_fields_path)


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    process_recordings()
    # local_data_test()


if __name__ == '__main__':
    main()