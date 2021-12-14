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

## Evaluation results
During evaluation process all results are gathered in `/model_dir/results.tsv`

Current metrics for rubert-base-cased: https://docs.google.com/spreadsheets/d/1385Bv8N05YWTGrPhPgq2KzIA9ltTm3bUEjUpLqas8jk/edit?usp=sharing

RSG submission link: https://russiansuperglue.com/login/submit_info/910

## RSG Tasks with jiant

| Task | Preprocessing |  Architecture | Sampling
| --- | ----------- | ------------------ | ------------
| LiDiRus | sentence1 + sentence2 | BERT + Linear layer(768,2)| Softmax
| RCB | premise + hypothesis |BERT + Linear layer(768,3)| Softmax
| PARus | premise + choice1 + choice2 | BERT + Linear layer(768, 2) | Softmax
| MuSeRC | text + questions + answers| BERT + Linear layer(768, 2) | Softmax
| TERRa | premise + hypothesis | BERT + Linear layer(768, 2) | Softmax
| RUSSE | sentence1+sentence2 | BERT + Linear layer(768, 2) | Softmax
| RWSD | text + sentencespan1 + sentencespan2 | BERT + ConvList + Attn + Linear layer(768, 2) | Softmax
| DaNetQA | text + question | BERT + Linear layer(768, 2) | Softmax
| RuCoS | text + query + entities |BERT + Linear layer(768, 2) | For every (tex+query+entity) triplet there is a tensor of logits (batch_size, 2). Then all logits are sorted by lambda x: x[1], and argmax(softmax)[:, -1] is taken. More info in RuCoSTask.get_metrics() and MultiTaskModel._multiple_choice_reading_comprehension_forward()
