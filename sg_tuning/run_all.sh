#!/bin/bash

git pull
python3 prepare.py

for task in superglue_broadcoverage_diagnostics cb copa multirc wic wsc boolq record superglue_winogender_diagnostics rte
do
    sbatch --job-name=SG-$task --output=SG-$task.out run.sh $task
done
