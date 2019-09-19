# a simple helper class to make working with snake input and output easier in a script
import os
from pathlib import Path

def makeFolders(output):
    # make folders used by the output varialbe if not exist
    # only needed when scripts are used independently
    # in snakemake, the folder will be created automatically
    # if the path contains '/' at the end, then it is considered as a directory and created accordingly

    for _,path in output.__dict__.items():
        folder = Path(path)
        if path[-1] =='/': #if this is a folder
            if not os.path.exists(folder):
                os.makedirs(folder)
                print('Created folder:' + str(folder))
        else:
            if not os.path.exists(folder.parent):
                os.makedirs(folder.parent)
                print('Created folder:' + folder.parent)

