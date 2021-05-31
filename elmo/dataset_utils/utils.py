import json
import time
import numpy as np
from string import punctuation
from pymorphy2 import MorphAnalyzer
from razdel import tokenize as razdel_tokenize
import tensorflow as tf
from tensorflow.keras import backend as K
from simple_elmo import ElmoModel


def keras_model(n_features=1024, MAXLEN=100, hidden_size=128, num_classes=2):
    """
        create a model to solve RSG tasks.
        params:
            n_features:     length of a single word embedding
                            received from simple_elmo model.get_elmo_vectors()
            MAX_LEN:    max sentence length in tokens, receieved from embeddings.shape[1]
            hidden_size: int
            n_label:    number of possible labels
    """
    if num_classes == 2:
        f_activation: str = 'sigmoid'
        loss: str = 'binary_crossentropy'
    else:
        f_activation: str = 'softmax'
        loss: str = 'categorical_crossentropy'

    embeddings = tf.keras.layers.Input(shape=(MAXLEN, n_features))

    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size, return_sequences=True, recurrent_dropout=0.1))(embeddings)

    pooling = tf.keras.layers.GlobalMaxPool1D()(lstm)
    dense = tf.keras.layers.Dense(hidden_size, activation='relu')(pooling)
    outputs = tf.keras.layers.Dense(
        num_classes, activation=f_activation)(dense)

    model = tf.keras.Model(inputs=embeddings, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    print(model.summary())

    return model


class RSG_MorphAnalyzer():

    def __init__(self):
        self.morpho = MorphAnalyzer()
        self.cashe = {}

    def normalize_sentences(self, sentences, use_lemmas=True):
        """
            receives a list of sentences
            returns list of lemmas by sentence
        """
        res = []
        for sentence in sentences:
            if use_lemmas:
                res.append(self.lemmatize(sentence))
            else:
                res.append(self.tokenize(sentence))

        return res

    def lemmatize(self, txt) -> list:
        """
            returns only lemmas
        """

        words = self.tokenize(txt)

        res = []

        for w in words:
            if w in self.cashe:
                res.append(self.cashe[w])
            else:
                r = self.morpho.parse(w)[0].normal_form
                res.append(r)
                self.cashe[w] = r

        return res

    def tokenize(self, txt) -> list:
        """
            tokenizes and removes punctuation from a string
        """
        punkt = punctuation + '«»—…–“”'
        tokens = []

        for word in list(razdel_tokenize(txt)):
            token = word.text.strip(punkt).lower()  # remove punctuation
            if token == "":  # skip empty elements if any
                continue
            tokens.append(token)

        return(tokens)


def save_output(data, path):
    """ a function to properly save the output before its submission """
    with open(path, mode="w") as file:
        for line in sorted(data, key=lambda x: int(x.get("idx"))):
            line["idx"] = int(line["idx"])
            file.write(f"{json.dumps(line, ensure_ascii=False)}\n")
