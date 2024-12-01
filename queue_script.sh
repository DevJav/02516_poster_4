#!/bin/sh
### ------------- specify queue name ---------------- 
###BSUB -q gpu02516i
###BSUB -q c02516
#BSUB -q gpua40


### ------------- specify job name ----------------
#BSUB -J aggregation

### ------------- specify gpu request---------------- 
#BSUB -gpu "num=1:mode=exclusive_process"

### ------------- specify number of cores ---------------- 
#BSUB -n 4 
#BSUB -R "span[hosts=1]"

### ------------- specify output file ----------------
#BSUB -o jobs-output/%Joutput.out
#BSUB -e jobs-output/%Jerror.out

### ------------- specify CPU memory requirements ---------------- 
#BSUB -R "rusage[mem=20GB]"

### ------------- maximum job execution time ----------------
#BSUB -W 01:00


source "/zhome/97/a/203937/idlcv/bin/activate"
python "earlyFusion.py"
