import time
import numpy as np
from string import punctuation
from pymorphy2 import MorphAnalyzer
from razdel import tokenize as razdel_tokenize
import tensorflow as tf
from tensorflow.keras import backend as K
from simple_elmo import ElmoModel


def keras_model(input_shape=512, hidden_size=128, num_classes=2):
    """
        create a model to solve RSG tasks.
        params:
            input_shape:     embeddings
                            received from simple_elmo model.get_elmo_vectors()
            MAX_LEN:    max sentence lenght in tokens
            n_label:    number of possible labels
            path_to_elmo: a path to a ready elmo model
    """
    if num_classes == 2:
        f_activation: str = 'sigmoid'
        loss: str = 'binary_crossentropy'
    else:
        f_activation: str = 'softmax'
        loss: str = 'categorical_crossentropy'

    # layers
    embeddings = tf.keras.layers.Input(shape=(input_shape,))
    # lstm = tf.keras.layers.LSTM(128, return_sequences=False)(embeddings)
    dense = tf.keras.layers.Dense(hidden_size, activation="relu")(embeddings)
    output = tf.keras.layers.Dense(num_classes, activation=f_activation)(dense)

    # the model
    model = tf.keras.Model(inputs=[embeddings], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    print(model.summary())

    return model


class RSG_MorphAnalyzer():

    def __init__(self):
        self.morpho = MorphAnalyzer()
        self.cashe = {}

    def lemmatize_sentences(self, sentences):
        """
            receives a list of sentences
            returns list of lemmas by sentence
        """
        res = []
        for sentence in sentences:
            res.append(self.lemmatize(sentence))

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
