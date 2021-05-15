# Rubert fine-tunining on RSG tasks

Based on https://github.com/RussianNLP/RussianSuperGLUE/tree/master/jiant-russian-v2

## Prepare Data and Model
You need to download RSG datasets and put them into `rubert_tuning/jiant-russian-v2/combined/` folder.

Model should be placed into `rubert_tuning/jiant-russian-v2/model/` direcrtory

## Simple Run `example_jiant_train.ipynb`

To fine-tune rubert on specifict task run the following commands:
```sh
cd ./jiant-russian-v2/
```
```sh
!chmod 755 ./scripts/russian-superglue-baselines.sh 
! ./scripts/russian-superglue-baselines.sh "russe"
```
Instead of russe it can be on of the values: 
"rcb", "parus", "muserc", "terra", "russe", "rwsd", "danetqa", "rucos"

## Metrics
During evaluation process all results are gathered in `/model_dir/results.tsv`

Current metrics for rubert-base-cased: https://docs.google.com/spreadsheets/d/1385Bv8N05YWTGrPhPgq2KzIA9ltTm3bUEjUpLqas8jk/edit?usp=sharing