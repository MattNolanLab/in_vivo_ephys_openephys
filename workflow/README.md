Each workflow contains a number of rules, each defined by its input, output, and script to execute. For general structure of the workflow file, please consult the official [snakemake documentation](https://snakemake.readthedocs.io/en/stable/tutorial/basics.html). The workflow file define all the input and output filename of the analysis. It is also easy to see which output files are created by which scripts.

The path of the scripts file are relative to the workflow file, i.e. you will need to use `../scripts/somescript.py` to refer to your script file in the workflow.

Always create a new workflow for a new experiment type rather than extending the old one with additional rules. The rule of thumb is that you first consider whether others with a different experimental design will be interested in your analysis. If not, then create a new workflow. You can always copy an old workflow file and rename to edit it.

The folder names of the input files accept wildcard. The wildcard is determined by the target file path, e.g. for a rule 

```
rule abc:
    input:
        infile = '{rootfolder}/input.txt'
    output:
        outfile = '{rootfolder}/output.txt'
```

if you execute `snakemake myfolder/output.txt`, then `rootfolder` will correspond to `myfolder`. 


Because snakemake needs to compare timestamp of input and output files to determine whether a particular rule needs to be run, so each rule should produce unique output files. 

You can also use a indicator file to enforce rule dependencies, e.g. by touching a file at the end of a rule
```
output:
    done = touch('{recording}/processed/workflow/plot_figures.done')
```
