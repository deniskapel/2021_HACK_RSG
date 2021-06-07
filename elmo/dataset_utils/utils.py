import json
from pymorphy2 import MorphAnalyzer
from razdel import tokenize as razdel_tokenize
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    GlobalMaxPool1D,
    Dense,
    LSTM)


def keras_model(n_features=1024, MAXLEN=100, hidden_size=16,
                num_classes=2, pooling=False, activation='softmax'):
    """
        create a model to solve RSG tasks.
        params:
            n_features:     length of a single word embedding
                            received from simple_elmo model.get_elmo_vectors()
            MAX_LEN:        max sentence length in tokens, receieved from embeddings.shape[1]
            hidden_size:    int
            n_classes:      number of possible labels
            pooling:        bool, apply pooling or not
    """
    if num_classes == 2:
        loss: str = 'binary_crossentropy'
    else:
        loss: str = 'categorical_crossentropy'

    embeddings = Input(shape=(MAXLEN, n_features), name="Elmo_embeddings")

    lstm = Bidirectional(
        LSTM(hidden_size, return_sequences=pooling),
        name='Bidirectional_LSTM')(embeddings)

    if pooling:
        lstm = GlobalMaxPool1D(name="Pooling1D")(lstm)

    outputs = Dense(num_classes, activation=activation, name="Output")(lstm)

    model = tf.keras.Model(inputs=embeddings, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


class RSG_MorphAnalyzer():

    def __init__(self):
        self.morpho = MorphAnalyzer()
        self.cashe = {}

    def normalize_sentences(self, sentences: list, use_lemmas: bool = True) -> list:
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

    def lemmatize(self, txt: str) -> list:
        """
            str -> tokens -> lemmas
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

    def tokenize(self, txt: str) -> list:
        """
            tokenizes a string uzing razdel
        """
        tokens = []

        for word in list(razdel_tokenize(txt)):
            token = word.text.lower()  # remove punctuation
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
