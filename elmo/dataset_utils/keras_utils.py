import re
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    GlobalMaxPool1D,
    Dense,
    LSTM,
    Lambda,
    Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from dataset_utils.global_vars import TIMESTAMP


def keras_model(n_features=1024, MAXLEN=100, hidden_size=256,
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


def keras_model_test(n_features=1024, max_lengths:list=[50,30], hidden_size=256,
                num_classes=2, pooling=False, activation='softmax'):
    """ test splitting input into several parts """
    if num_classes == 2:
        loss: str = 'binary_crossentropy'
    else:
        loss: str = 'categorical_crossentropy'

    MAXLEN = sum(max_lengths)

    embeddings = Input(shape=(MAXLEN, n_features), name="Elmo_embeddings")

    inner_idx = 0
    all_parts = []
    
    for i in range(len(max_lengths)):
        outer_idx = inner_idx+max_lengths[i]
        part = Lambda(lambda x: x[:,inner_idx:outer_idx,:])(embeddings)
        inner_idx = outer_idx

        test_lstm = Bidirectional(
            LSTM(hidden_size, return_sequences=pooling))(part)

        if pooling:
            test_lstm = GlobalMaxPool1D()(test_lstm)

        all_parts.append(test_lstm)

    all_parts = tf.concat(all_parts, axis=1)

    outputs = Dense(num_classes, activation=activation, name="Output")(all_parts)

    model = tf.keras.Model(inputs=embeddings, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
