import os
def delete_files_with_extension(path_to_folder, extension):
    # eg. delete_files_with_extension(path_to_folder = "/mnt/datastore/Harry/Cohort8_may2021\of/", extension = ".spikes")
    inputpath = path_to_folder

    for dirpath, dirnames, filenames in os.walk(inputpath):
        for name in filenames:
            if os.path.join(dirpath, name).endswith(extension):
                print("I want to delete this file:, " +os.path.join(dirpath, name))
                # os.remove(os.path.join(dirpath, name))
    print("I hope this worked")
    # this function actually works

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    path_to_folder = "/mnt/datastore/Teris/FragileX"
    delete_files_with_extension(path_to_folder, extension=".spikes")


if __name__ == '__main__':
    main()