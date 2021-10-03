import json

from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from dataset_utils.global_vars import DTYPE


def save_output(data, path):
    """ a function to properly save the output before its submission """
    with open(path, mode="w") as file:
        for line in sorted(data, key=lambda x: int(x.get("idx"))):
            line["idx"] = int(line["idx"])
            file.write(f"{json.dumps(line, ensure_ascii=False)}\n")

TEXT_FIELDS = {
    "DaNetQA": ["question", 'passage'],
    'LiDiRus': ['sentence1', 'sentence2'],
    "MuSeRC": [], # hard coded due to a complex structure
    "PARus": ['premise', 'choice1', 'choice2'],
    'RCB': ['premise', 'hypothesis'],
    "RuCoS": [], # hard coded due to a complex structure
    'RUSSE': ['sentence1', 'sentence2'],
    'RWSD': ['text'],
    'TERRa': ['premise', 'hypothesis']
}

class DataGenerator(Sequence):
    """
        Generates data for TF Keras to work with huge datasets
        The example comes from here 
        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(
        self, samples: list, labels: list, elmo_model, max_lengths: list,
        batch_size=32, n_classes=2, shuffle=True):
        self.max_lengths = max_lengths # maximum lengths for each part of the sample
        self.batch_size = batch_size
        self.x = samples # [list_of_samples_of_part_1..list_of_samples_of_part_n]
        self.y = labels
        self.elmo_model = elmo_model
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        """
            Updates indexes after each epoch
            Shuffling the order so that batches between epochs do not look alike. 
            It can make a model more robust.
        """
        # features consist of several parts so calc size using labels
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """
            Denotes the number of batches per epoch

            A common practice is to set this value to [num_samples / batch size⌋
            so that the model sees the training samples at most once per epoch.
        """
        # features consist of several parts so calc size using labels
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data"""
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = [[part[i] for i in indexes] for part in self.x]
        batch_y = [self.y[i] for i in indexes]

        # Generate embeddings and labels for a batch
        X, y = self.__data_generation(batch_x, batch_y)

        return X, y

    def __data_generation(self, batch_x: list, batch_y: list):
        """
            Generates data containing batch_size samples
            X : (
                n_samples, max_sample_length, elmo_vector_size, n_channels
                )
        """ 
        X = self.__get_embeddings(batch_x)
        # y = to_categorical(batch_y, num_classes=self.n_classes)

        return X, to_categorical(batch_y, num_classes=self.n_classes)


    def __get_embeddings(self, batch_x: list) -> np.ndarray:
        """ extract embeddings for each part and stack them int"""
        # extract embeddings
        embeddings = [self.elmo_model.get_elmo_vectors(part) for part in batch_x]
        # pad embeddings based on a max_lengths in a train set
        embeddings = [pad_sequences(d, maxlen=l, dtype=DTYPE, padding='post')
                         for d, l in zip(embeddings, self.max_lengths)]

        # merge sample parts into a complete samples and return them
        return np.hstack(embeddings)
