import pandas as pd
from string import punctuation
from pymorphy2 import MorphAnalyzer
from razdel import tokenize as razdel_tokenize
import tensorflow as tf
from tensorflow.keras import backend as K
from simple_elmo import ElmoModel


def build_model(MAX_LEN: int,
                VOCAB_SIZE: int,
                n_labels: int,
                path_to_elmo: str):
    """
        create a model to solve RSG tasks.
        params:
            embeddings:     embeddings
                            received from simple_elmo model.get_elmo_vectors()
            MAX_LEN:    max sentence lenght in tokens
            n_label:    number of possible labels
            path_to_elmo: a path to a ready elmo model
    """
    if n_labels == 2:
        f_activation: str = 'sigmoid'
        loss: str = 'binary_crossentropy'
    else:
        f_activation: str = 'softmax'
        loss: str = 'categorical_crossentropy'

    # layers - does not work
    embeddings = tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE, output_dim=64, input_length=MAX_LEN)
    lstm = tf.keras.layers.LSTM(128, return_sequences=False)(embeddings)
    dense = tf.keras.layers.Dense(64, activation='relu')(lstm)
    outputs = tf.keras.layers.Dense(n_labels, activation=f_activation)(dense)

    # model
    model = tf.keras.Model(inputs=embeddings, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


class RSG_MorphAnalyzer():

    def __init__(self):
        self.morpho = MorphAnalyzer()
        self.cashe = {}

    def lemmatize_sentences(self, sentences):
        """
            receives a list of tokens by sentence
            returns list of lemmas by sentence
        """
        res = []
        for sentence in sentences:
            res.append(self.lemmantize(sentence))

        return(res)

    def lemmantize(self, txt) -> list:
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

        return(res)

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


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
