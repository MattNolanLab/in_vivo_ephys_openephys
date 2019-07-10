# in_vivo_ephys_openephys
Analysis for in vivo electrophysiology recordings saved in open ephys format. 

The current pipeline runs on a linux computer and uses MountainSort 3 to automatically sort data. The analysis is set up to minimize user interaction as much as possible, and analyses are initiated by copying files to a designated computer.

Sorted clusteres are automatically curated and output plots of spatial firing properties are generated. Spatial scores (grid score, HD score, speed score) are calculated and saved in pickle (pandas) data frames.

Please see wiki for more detailed information https://github.com/MattNolanLab/in_vivo_ephys_openephys/wiki/Pipeline-overview
