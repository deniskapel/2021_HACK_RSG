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


def infer_embeddings(texts, elmo, warmup=True, layers="average"):
    """ 
        uses simple elmo and a ready embedding
        to infer embeddings from a given set of sentences
        params:
            texts: list of list of tokens (lemmas)
            elmo: ElmoModel
            layers: average (default) 
                        - return the average of all ELMo layers for each word;
                    top: return only the top (last) layer for each word;
                    all: return all ELMo layers for each word 
                        - (an additional dimension appears in the produced tensor,
                        with the shape equal to the number of layers in the model,
                        3 as a rule)
        output: tensor
    """
    start = time.time()
    elmo_vectors = elmo.get_elmo_vectors(texts, warmup=warmup, layers=layers)

    nr_words = len([item for sublist in texts for item in sublist])

    feature_matrix = np.zeros((nr_words, elmo.vector_size))
    row_nr = 0
    for vect, sent in zip(elmo_vectors, texts):
        cropped_matrix = vect[: len(sent), :]
        for row in cropped_matrix:
            feature_matrix[row_nr] = row
            row_nr += 1

    end = time.time()
    processing_time = int(end - start)

    print(
        f"ELMo embeddings for your input are ready in {processing_time} seconds")

    return feature_matrix
