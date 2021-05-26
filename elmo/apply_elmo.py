import sys
import random as python_random
import numpy as np
from collections import Counter
from dataset_utils.utils import RSG_MorphAnalyzer, keras_model
from simple_elmo import ElmoModel
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from dataset_utils.features import build_features
from sklearn.metrics import classification_report


def main(args):
    if len(args) != 2:
        sys.stderr.write(
            'Usage: test_elmo.py <path_to_dataset_folder> <path_to_elmo_model>\n')
        sys.exit(1)

    PATH_TO_DATASET = args[0]
    PATH_TO_ELMO = args[1]

    morph = RSG_MorphAnalyzer()

    # load data
    train, _ = build_features('%strain.jsonl' % (PATH_TO_DATASET))
    val, _ = build_features('%sval.jsonl' % (PATH_TO_DATASET))
    # test, ids = build_features('%stest.jsonl' % (PATH_TO_DATASET))

    # preprocess data
    X_train = morph.normalize_sentences(train[0], lemmas=True)
    y_train = train[1]
    classes = sorted(list(set(y_train)))
    num_classes = len(classes)
    y_train = [classes.index(i) for i in y_train]
    y_train = to_categorical(y_train, num_classes)
    X_valid = morph.normalize_sentences(val[0], lemmas=True)
    y_valid = val[1]
    y_valid = [classes.index(i) for i in y_valid]
    y_valid = to_categorical(y_valid, num_classes)

    # get embeddings
    elmo = ElmoModel()
    elmo.load(PATH_TO_ELMO, max_batch_size=64)
    X_train_embeddings = elmo.get_elmo_vectors(X_train[0:100])
    X_val_embeddings = elmo.get_elmo_vectors(X_valid[0:100])

    n_samples, max_len, n_features = X_train_embeddings.shape

    # initialize a keras model that takes elmo embeddings as its input
    model = keras_model(n_features=n_features,
                        hidden_size=128,
                        num_classes=num_classes)

    earlystopping = EarlyStopping(
        monitor="val_accuracy", min_delta=0.0001, patience=2, verbose=1, mode="max"
    )

    # Train the compiled model on the training data
    model.fit(
        X_train_embeddings,
        y_train[0:100],
        epochs=5,
        validation_data=(X_val_embeddings, y_valid[0:100]),
        batch_size=32,
        callbacks=[earlystopping],
    )

    preds = model.predict(X_val_embeddings)

    print(preds.shape)

    print(preds)
    # # map predictions to the binary {0, 1} range:
    preds = np.around(preds)
    print(preds.shape)
    print(preds)
    # # Convert predictions from integers back to text labels:
    preds = [classes[int(np.argmax(pred))] for pred in preds]
    print(len(preds))
    print(preds)

    print(classification_report(val[1][0:100], preds))


if __name__ == '__main__':
    # For reproducibility:
    # np.random.seed(42)
    # python_random.seed(42)
    # tf.random.set_seed(42)

    main(sys.argv[1:])
