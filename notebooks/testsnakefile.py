#%%
import snakemake

#%%
output = snakemake.snakemake('vr_workflow.smk',dryrun=True)
#%%
c=snakemake.snakemake('vr_workflow.smk',dryrun=True,
    targets=['testData/M1_D31_2018-11-01_12-28-25_short/processed/results.txt'],
    summary=True)

#%%
snakemake.snakemake('vr_workflow.smk',dryrun=True,
    targets=['testData/M1_D31_2018-11-01_12-28-25_short/processed/results.txt'],
    detailed_summary=True)

#%%
snakemake.snakemake('vr_workflow.smk',dryrun=True,
    targets=['testData/M1_D31_2018-11-01_12-28-25_short/processed/results.txt'],
    print_compilation=True)

#%%
snakemake.snakemake('vr_workflow.smk',dryrun=True,
    targets=['testData/M1_D31_2018-11-01_12-28-25_short/processed/results.txt'],
    list_code_changes=True)

#%%
snakemake.snakemake('vr_workflow.smk',dryrun=True,
    targets=['testData/M1_D31_2018-11-01_12-28-25_short/processed/results.txt'],
    list_input_changes=True)

#%%
snakemake.snakemake('vr_workflow.smk',dryrun=True,
    targets=['testData/M1_D31_2018-11-01_12-28-25_short/processed/results.txt'],
    printdag=True) 

#%%
snakemake.snakemake('vr_workflow.smk',dryrun=True,
    targets=['testData/M1_D31_2018-11-01_12-28-25_short/processed/results.txt'],
    printfilegraph=True, forceall=True)     

#%%
from SnakeIOHelper import IOParser
import snakemake

parser = IOParser('vr_workflow.smk', ['testData/M1_D31_2018-11-01_12-28-25_short/processed/results.txt'])
snakeIO = parser.getInputOutput4rule('process_expt')
print(snakeIO.input.spatial_firing)
#%%
snakemake.logger.setup_logfile()
snakefile = 'vr_workflow.smk'
workflow = snakemake.Workflow(snakefile,default_resources=None)
workflow.include(snakefile)
workflow.check()
workflow.execute(dryrun=True, updated_files=[], summary=True, 
    targets=['testData/M1_D31_2018-11-01_12-28-25_short/processed/results.txt'])

#%%
dag = workflow.persistence.dag

#%%
def getJobList(dag):
    jobs = {}
    for j in dag.jobs:
        jobs[j.name] = j
    return jobs

jobs = getJobList(dag)



#%%
import os
os.supports_follow_symlinks

#%%
