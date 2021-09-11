# This folder is about ELMo embeddings applied to solve the Russian SuperGLUE tasks


Scores on [validate](https://docs.google.com/spreadsheets/d/1BBArqZo6pk1lnP2-KlMjHnSPKvPOJfBb4qqC07Ao5ps/edit?usp=sharing)

1. To start, download data.

```bash
   make data
```

The output is folder with all RSG datasets.


2. The [NLPL vector repository](http://vectors.nlpl.eu/repository/) containts several ready ELMo embeddings that were used for testing. Additionally, it will download a UD model that is used for preprocessing down the pipeline,

```bash
   make models
```

`Simple_elmo` - the library that is used to apply ELMo to text files. Its documentation is available [here](https://github.com/ltgoslo/simple_elmo).


3. Before vectorizing, it is necessary to preprocess the text fields in each sample. Otherwise, text fields will be split on spaces only in the following way: `["This", "is", "it."]`. To preprocess, run

```python
   python3 preprocessing.py
```

Preprocessing is either tokenization (default) or lemmatisation (add `--lemmas` to the command above). The result is saved into `/data` with all text fields ready to be split using spaces only, but properly. Preprocessing takes time.

4. To generate submission for a dataset, use e.g.

```bash
   python3 apply_elmo.py -t combined/TERRa/ -e models/taiga/
```

Additional arguments:
         
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