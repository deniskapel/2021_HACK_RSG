import sys
import random as python_random
import numpy as np
from collections import Counter
from simple_elmo import ElmoModel
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from dataset_utils.utils import RSG_MorphAnalyzer, keras_model, save_output
from dataset_utils.muserc import get_MuSeRC_predictions, MuSeRC_metrics
from dataset_utils.features import build_features


def main(args):
    if len(args) != 3:
        sys.stderr.write(
            'Usage: test_elmo.py <path_to_dataset_folder> <path_to_elmo_model> <use lemmas or tokens>\n')
        sys.exit(1)

    PATH_TO_DATASET = args[0]
    PATH_TO_ELMO = args[1]
    OUTPUT_DIR = 'submissions/'
    # get a dataset name
    PATH_TO_OUTPUT = '%s%s.jsonl' % (OUTPUT_DIR, PATH_TO_DATASET[9:-1])

    if args[2] not in ['lemmas', 'tokens']:
        sys.stderr.write(
            'Last arguments has to be either lemmas or tokens, please specify.\n')
        sys.exit(1)
    elif args[2] == 'lemmas':
        USE_LEMMAS = True
    else:
        USE_LEMMAS = False

    morph = RSG_MorphAnalyzer()

    # load data
    train, _ = build_features('%strain.jsonl' % (PATH_TO_DATASET))

    val, _ = build_features('%sval.jsonl' % (PATH_TO_DATASET))
    if PATH_TO_DATASET[9:-1] not in ['MuSeRC', "RuCoS"]:
        test, ids = build_features('%stest.jsonl' % (PATH_TO_DATASET))

    # preprocess data
    X_train = morph.normalize_sentences(train[0], use_lemmas=USE_LEMMAS)
    X_valid = morph.normalize_sentences(val[0], use_lemmas=USE_LEMMAS)
    classes = sorted(list(set(train[1])))
    y_train = [classes.index(i) for i in train[1]]
    num_classes = len(classes)
    y_train = to_categorical(y_train, num_classes)
    y_valid = [classes.index(i) for i in val[1]]
    y_valid = to_categorical(y_valid, num_classes)

    # get embeddings
    elmo = ElmoModel()
    elmo.load(PATH_TO_ELMO, max_batch_size=64)
    X_train_embeddings = elmo.get_elmo_vectors(X_train[0:10])
    _, MAXLEN, n_features = X_train_embeddings.shape
    X_val_embeddings = elmo.get_elmo_vectors(X_valid[0:10])
    # shape val based on train dimensions
    X_val_embeddings = tf.keras.preprocessing.sequence.pad_sequences(
        X_val_embeddings, maxlen=MAXLEN)

    # initialize a keras model that takes elmo embeddings as its input
    model = keras_model(n_features=n_features,
                        MAXLEN=MAXLEN,
                        hidden_size=128,
                        num_classes=num_classes)

    earlystopping = EarlyStopping(
        monitor="val_accuracy", min_delta=0.0001, patience=2, verbose=1, mode="max"
    )

    # Train the compiled model on the training data
    model.fit(
        X_train_embeddings,
        y_train[0:10],
        epochs=5,
        validation_data=(X_val_embeddings, y_valid[0:10]),
        batch_size=64,
        callbacks=[earlystopping])

    if 'MuSeRC' in PATH_TO_DATASET:
        preds, labels, _ = get_MuSeRC_predictions(
            '%sval.jsonl' % (PATH_TO_DATASET), MAXLEN,
            elmo, model,  # elmo, keras
            morph, use_lemmas=USE_LEMMAS  # pymoprhy2
        )
        # check on validation
        em, f1a = MuSeRC_metrics(preds, labels)
        print("Evaluation of val: em is %s, f1a is %s" % (em, f1a))

        _, _, res = get_MuSeRC_predictions(
            '%stest.jsonl' % (PATH_TO_DATASET), MAXLEN,
            elmo, model,  # elmo, keras
            morph, use_lemmas=USE_LEMMAS  # pymoprhy2
        )
        save_output(res, PATH_TO_OUTPUT)
    else:
        preds = model.predict(X_val_embeddings)
        # map predictions to the binary {0, 1} range:
        # Convert predictions from integers back to text labels:
        preds = [classes[int(np.argmax(pred))] for pred in np.around(preds)]
        # Check on validation
        print(classification_report(val[1][0:10], preds))

        # Preprocess test sets
        X_test = morph.normalize_sentences(test[0], use_lemmas=USE_LEMMAS)
        X_test_embeddings = elmo.get_elmo_vectors(X_test[0:10])
        # shape val based on train dimensions
        X_test_embeddings = tf.keras.preprocessing.sequence.pad_sequences(
            X_test_embeddings, maxlen=MAXLEN)

        preds = model.predict(X_test_embeddings)
        preds = [classes[int(np.argmax(pred))] for pred in np.around(preds)]
        preds = [
            {"idx": i, "label": str(label).lower()} for i, label in zip(ids, preds)]
        save_output(preds, PATH_TO_OUTPUT)


if __name__ == '__main__':
    # For reproducibility:
    np.random.seed(42)
    python_random.seed(42)
    tf.random.set_seed(42)

    main(sys.argv[1:])
