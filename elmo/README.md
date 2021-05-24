# This folder is about ELMo embeddings applied to solve the Russian SuperGLUE tasks

1. To start, download data.

```bash
    make data
```

The output is /combined folder with all RSG datasets.

1.1. (Optional). The [NLPL vector repository](http://vectors.nlpl.eu/repository/) containts several ready ELMo embeddings.

```bash
   make models
```

This command will download two models: `Russian Wikipedia dump of December 2018 + Russian National Corpus` (non-lemmatized) and `Taiga corpus` (lemmatized). They will be stored in the ./models folder and named 195.zip and 199.zip respectively. To use either of these models with simple **Simple_elmo**, add the path to it via PATH_TO_ELMO.

```python
   from simple_elmo import ElmoModel
   model = ElmoModel()
   model.load(PATH_TO_ELMO)
```
     
2. [LiDiRus dataset](combined/LiDiRus) is a diagnostic dataset. To apply feature extraction to it, rename the LiDiRus.jsonl file to test.jsonl, and add train.jsonl and val.jsonl from [TERRa](combined/TERRa) dataset to the LiDiRus folder