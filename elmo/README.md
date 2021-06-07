# This folder is about ELMo embeddings applied to solve the Russian SuperGLUE tasks


Scores on [validate](https://docs.google.com/spreadsheets/d/1BBArqZo6pk1lnP2-KlMjHnSPKvPOJfBb4qqC07Ao5ps/edit?usp=sharing)

1. To start, download data.

```bash
    make data
```

The output is folder with all RSG datasets.

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

2. To generate submission for a dataset, use e.g.

```bash
   python3 apply_elmo.py -t combined/TERRa/ -e models/taiga/
```

Additional arguments:

> --lemmas

if added, lemmatization will be applied. By default, basic tokenisation. Puntcuation is kept as an individual token. 
         
> --pooling

if added, pooling will be added after LSTM and before logreg. By default, only the last layer is used for logreg.

> --elmo_layers average

Choose between average, all or top. See Simple Elmo [documentation](https://github.com/ltgoslo/simple_elmo).
    
> --activation sigmoid

Choose between sigmoid and softmax. By default, softmax is applied for an activation function.

> --num_epochs 15

Default is 15. Keras epochs

> --hidden_size 16

Default is 16. Keras hidden_size.

> --args.batch_size 32

Default is 32. Elmo and Keras batch_size.

For now, only **LiDiRus, TERRa, RCB, and RUSSE** are fully ready, DaNetQA and RUSSE require more resources to compute and RWSD needs some revision.

3. Logs are stored in the [here](/logs). Each run will generate a TASKNAME_TIME.log file