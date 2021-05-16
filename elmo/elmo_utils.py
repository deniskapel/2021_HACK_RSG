# /bin/env python3
import tensorflow as tf
from simple_elmo import ElmoModel


def build_model(MAX_LEN: int,
                n_labels: int = 2):
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

    # layers
    inputs = tf.keras.layers.Input(shape=(MAX_LEN,))
    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    dense = tf.keras.layers.Dense(64, activation='relu')(lstm)
    outputs = tf.keras.layers.Dense(n_labels, activation=f_activation)(dense)

    # model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model
