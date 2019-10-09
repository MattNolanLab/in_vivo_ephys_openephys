# a simple helper class to make working with snake input and output easier in a script
import os
from pathlib import Path
import snakemake
import sys

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
                print('Created folder:' + str(folder.parent))

def getSnake(snakefile:str, targets:list, rule:str):
    # determine the running environment and return the snake object appropriately
    parser = IOParser(snakefile, targets)
    io = parser.getInputOutput4rule(rule)
    return io

class IOParser:
    def __init__(self, snakefile:str, targets:list):
        self.snakefile = snakefile
        self.targets = targets

        self.workflow = self.compileWorkflow()
        self.dag = self.workflow.persistence.dag


    def compileWorkflow(self):
        snakemake.logger.setup_logfile()
        snakefile = 'vr_workflow.smk'
        workflow = snakemake.Workflow(snakefile,default_resources=None)
        workflow.include(self.snakefile)
        workflow.check()

        workflow.execute(dryrun=True, updated_files=[], quiet=True,
            targets=self.targets)

        return workflow

    def getInputOutput(self):
        return self.getJobList(self.dag)

    def getInputOutput4rule(self,rulename:str):
        io = self.getInputOutput()
        return io[rulename]
    
    def getJobList(self,dag):
        jobs = {}
        for j in dag.jobs:
            jobs[j.name] = j
        return jobs


