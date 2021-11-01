import re
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    GlobalMaxPool1D,
    Dense,
    Layer,
    LSTM,
    Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from dataset_utils.global_vars import TIMESTAMP


def keras_model(n_features=1024, size_splits:list =[10,10], hidden_size=256,
            num_classes=2, pooling=False, activation='softmax'):
    """
    create a model to solve RSG tasks.
    params:
    n_features:     length of a single word embedding
                    received from simple_elmo model.get_elmo_vectors()
    size_split:     list of max lenghts for each sample parts
    hidden_size:    int
    n_classes:      number of possible labels
    pooling:        bool, apply pooling or not
    """
    if num_classes == 2:
        loss: str = 'binary_crossentropy'
    else:
        loss: str = 'categorical_crossentropy'

    seq_length = sum(size_splits)
    embeddings = Input(shape=(seq_length, n_features), name="Elmo_embeddings")
    parts = SplitLayer(size_splits)(embeddings)
    merged = []
    
    for i in range(len(size_splits)):
        lstm = Bidirectional(
            LSTM(hidden_size, return_sequences=pooling))(parts[i])
        if pooling:
            lstm = GlobalMaxPool1D()(lstm)

        merged.append(lstm)

    merged = tf.keras.layers.Concatenate(axis=1)(merged)
    outputs = Dense(num_classes, activation=activation, name="Output")(merged)
    model = tf.keras.Model(inputs=embeddings, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


class SplitLayer(tf.keras.layers.Layer):
    """ 
    A class splitting embeddings into parts, e.g. premise and question 
    Number of parts depends on the number of sizes of parts.
    """
    def __init__(self, size_splits: list):
        super(SplitLayer, self).__init__()
        self.size_splits = size_splits

    def call(self, inputs):
        return tf.split(inputs, num_or_size_splits=self.size_splits, axis=1)


# CALLBACKS
early_stopping = EarlyStopping(
    monitor='val_accuracy', min_delta=0.001,
    patience=3, verbose=1, mode='max'
)


def wrap_checkpoint(model_name: str):
    path = f'checkpoints/{model_name}'
    Path(path).mkdir() 


    return ModelCheckpoint(
        filepath = path + '/checkpoint',
        monitor='val_accuracy', verbose=1,
        save_weights_only=True,
        save_best_only=True,
        mode='max', save_freq='epoch'
    )

