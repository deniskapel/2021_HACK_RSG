import json
import math
import logging
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import time
import numpy as np
import simple_elmo

from dataset_utils.elmo_utils import reshape4Dto3D

def save_output(data, path):
    """ a function to properly save the output before its submission """
    with open(path, mode="w") as file:
        for line in sorted(data, key=lambda x: int(x.get("idx"))):
            line["idx"] = int(line["idx"])
            file.write(f"{json.dumps(line, ensure_ascii=False)}\n")


class DataGenerator(Sequence):
    """
        Generates data for TF Keras to work with huge datasets
        The example comes from here
        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(
            self, samples: list, labels: list, elmo_model, elmo_graph, n_features,
            max_lengths: list, batch_size=32, n_classes=2, shuffle=True, layers="top"):
        self.max_lengths = max_lengths  # maximum lengths for each part of the sample
        self.batch_size = batch_size
        self.x = samples  # [list_of_samples_of_part_1..list_of_samples_of_part_n]
        self.y = labels
        self.indexes = np.arange(len(self.y))
        self.elmo_model = elmo_model
        self.elmo_graph = elmo_graph
        self.n_features = n_features
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dim2 = sum(max_lengths)
        self.layers = layers
        logging.basicConfig(
            format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)

        with self.elmo_graph.as_default() as current_graph:
            self.tf_session = tf.compat.v1.Session(graph=self.elmo_graph)
            with self.tf_session.as_default() as sess:
                self.elmo_model.elmo_sentence_input = simple_elmo.elmo.weight_layers(
                    "input", self.elmo_model.sentence_embeddings_op, use_layers=self.layers)
                sess.run(tf.compat.v1.global_variables_initializer())

        self.logger.info("Generator initialized")

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        Shuffling the order so that batches between epochs do not look alike.
        It can make a model more robust.
        """
        # features consist of several parts so calc size using labels
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch

        A common practice is to set this value to [num_samples / batch size⌋
        so that the model sees the training samples at most once per epoch.
        """
        # features consist of several parts so calc size using labels
        return math.ceil(len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        """Generate one batch of data"""
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [[part[i] for i in indexes] for part in self.x]
        batch_y = [self.y[i] for i in indexes]

        # Generate embeddings and labels for a batch
        x, y = self.__data_generation(batch_x, batch_y)

        return x, y

    def __data_generation(self, batch_x: list, batch_y: list):
        """
        Generates data for training, validation and prediction.
        X: (n_samples, sum(self.max_lengths), self.elmo_model.vector_size)
        y: one-hot encoded labels
        """
        x = self.__get_embeddings(batch_x)
        y = to_categorical(batch_y, self.n_classes)

        return x, y

    def __get_embeddings(self, batch_x: list) -> np.ndarray:
        """ extract embeddings for each part and stack them int"""
        # last batch can be smaller than global batch_size
        # get current batch size using one of the sample parts
        current_bs = len(batch_x[0])
        # number of features depends on the number of layers returned by elmo
        embeddings = np.zeros((current_bs, self.dim2, self.n_features))

        start = time.time()
        self.logger.debug(f"OK, generating embeddings for {np.shape(embeddings)}...")

        # set a lower boundary for indexing the output array
        lower_id = 0
        for d, l in zip(batch_x, self.max_lengths):
            e = self.elmo_model.get_elmo_vectors(d, warmup=False,
                                                 layers=self.layers,
                                                 session=self.tf_session)

            if self.layers == 'all':
                e =  reshape4Dto3D(e)

            # truncate dim2 if it is larger than l
            e = e[:,:l,:]

            # get the upper boundary for numpy indexing
            # it avoids stacking and padding but truncates whenever necessary
            upper_id = lower_id + e.shape[1]

            # store part embeddings into the output matrix
            embeddings[:, lower_id:upper_id, :] = e
            # reset a lower boundary
            lower_id = lower_id + l

        end = time.time()
        processing_time = int(end - start)
        self.logger.debug(f"It took {processing_time} seconds")

        # merge sample parts into a complete samples and return them
        return embeddings
