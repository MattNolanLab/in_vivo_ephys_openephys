import glob
import os
import PostSorting.open_field_spatial_data

server_path = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Tizzy/Cohorts1-2/'


def process_axona_recordings():
    axona_folder = False
    for dir, sub_dirs, files in os.walk(server_path):
        os.path.isdir(dir)
        for subdir in sub_dirs:
            for file in glob.glob(dir + '/' + subdir + '*'):
                axona_files = glob.glob(os.path.join(file, '*.set'))
                if len(axona_files) > 0:
                    axona_folder = True

                    if axona_folder:
                        try:
                            path_to_position_file, is_found = PostSorting.open_field_spatial_data.find_axona_position_file(file + '/')
                            position_data = PostSorting.open_field_spatial_data.read_position_axona(path_to_position_file)
                            position_data.to_pickle(file + '/axona_position.pkl')
                        except:
                            print('did not manage to process ' + file)


def main():
    process_axona_recordings()


if __name__ == '__main__':
    main()

