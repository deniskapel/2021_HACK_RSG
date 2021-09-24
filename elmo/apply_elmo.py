import argparse
import logging
import time
import sys
import re
import random as python_random

import numpy as np
from simple_elmo import ElmoModel
# import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from sklearn.metrics import classification_report
# from sklearn.metrics import matthews_corrcoef

# from dataset_utils.utils import save_output
# from dataset_utils.keras_utils import keras_model
from dataset_utils.features import build_features


def main(
    path_to_task: str, task_name_first_char: int, path_to_elmo: str,
    elmo_layers: str, pooling: bool, activation: str, epochs: int, 
    hidden_size: int, batch_size: int):

    TASK_NAME = path_to_task[task_name_first_char:-1]
    INPUT_FOLDER = path_to_task[:task_name_first_char]

    if TASK_NAME in ['MuSeRC', "RuCoS"]:
        sys.stderr.write(
            'Use elmo_to_rucos_and_muserc.py for this task\n')
        sys.exit(1)

    PATH_TO_OUTPUT = 'submissions/%s.jsonl' % (TASK_NAME)

    """ DATA """
    if TASK_NAME == 'LiDiRus':
        train, _ = build_features('%sTERRa/train.jsonl' % (INPUT_FOLDER))
        val, _ = build_features('%sTERRa/val.jsonl' % (INPUT_FOLDER))
        test, ids = build_features('%sLiDiRus.jsonl' % (path_to_task))
    else:
        train, _ = build_features('%strain.jsonl' % (path_to_task))
        val, _ = build_features('%sval.jsonl' % (path_to_task))
        test, ids = build_features('%stest.jsonl' % (path_to_task))

    # extract samples from sample+label bundles
    X_train = list(zip(*train[0]))
    X_valid = list(zip(*val[0]))

    # tokenize each sample in a set
    X_train = [[sample.split() for sample in part] for part in X_train]
    X_valid = [[sample.split() for sample in part] for part in X_valid]

    """ EMBEDDINGS """
    logger.info(f"=======================")
    logger.info(f"loading Elmo model")
    elmo = ElmoModel()
    elmo.load(path_to_elmo, max_batch_size=batch_size)

    # create train embeddings    
    X_train_embeddings = [elmo.get_elmo_vectors(
        part, layers=elmo_layers) for part in X_train]

    # get max_length for each sample part, 
    # e.g. 7 for premise and 5 for hypothesis 
    max_lengths = [part.shape[1] for part in X_train_embeddings]
    
    X_train_embeddings = np.hstack(tuple(X_train_embeddings))

    # create validate embeddings
    X_val_embeddings = [elmo.get_elmo_vectors(
        part, layers=elmo_layers) for part in X_valid]

    # Dtype for padding, otherwise rounded to int32
    DTYPE = X_train_embeddings.dtype
    # add padding before each sentence using train maxlength
    X_val_embeddings = [pad_sequences(d, maxlen=l, dtype=DTYPE)
                        for d, l in zip(X_val_embeddings, max_lengths)]

    X_val_embeddings = np.hstack(tuple(X_val_embeddings))

    del X_train, X_valid

    # reshape labels
    classes = sorted(list(set(train[1])))
    y_train = [classes.index(i) for i in train[1]]
    num_classes = len(classes)
    y_train = to_categorical(y_train, num_classes)
    y_valid = [classes.index(i) for i in val[1]]
    y_valid = to_categorical(y_valid, num_classes)

    _, MAXLEN, n_features = X_train_embeddings.shape
    logger.info(f"=======================")
    logger.info(f'Tensor_shape {X_train_embeddings.shape}')
    logger.info(f"=======================")


    """ MODEL """
    # initialize a keras model that takes elmo embeddings as its input
    model = keras_model(n_features=n_features,
                        MAXLEN=MAXLEN,
                        hidden_size=hidden_size,
                        num_classes=num_classes,
                        pooling=pooling,
                        activation=activation)

    model.summary(print_fn=logger.info)

    logger.info(f"Start training")
    logger.info("====================")
    model.fit(
        X_train_embeddings,
        y_train,
        epochs=epochs,
        validation_data=(X_val_embeddings, y_valid),
        batch_size=batch_size)

    del X_train_embeddings, train


    """ PREDICTING """
    logger.info("====================")
    logger.info("Start predicting.")
    preds = model.predict(X_val_embeddings)
    preds = [classes[int(np.argmax(pred))] for pred in preds]
    # Log score on validation
    if TASK_NAME == 'LiDiRus':
        logger.info(f"MC score is {matthews_corrcoef(val[1], preds)}")
    else:
        logger.info(classification_report(val[1], preds))

    del X_val_embeddings, val


    """ APPLY TO TEST """

    # Preprocess the test split
    X_test = list(zip(*test[0]))
    X_test = [morph.normalize_sentences(
        part, use_lemmas=use_lemmas) for part in X_test]
    X_test_embeddings = [elmo.get_elmo_vectors(
        part, layers=elmo_layers) for part in X_test]
    X_test_embeddings = [pad_sequences(d, maxlen=l)
                         for d, l in zip(X_test_embeddings, max_lengths)]
    X_test_embeddings = np.hstack(tuple(X_test_embeddings))

    preds = model.predict(X_test_embeddings)
    preds = [classes[int(np.argmax(pred))] for pred in np.around(preds)]
    preds = [
        {"idx": i, "label": str(label).lower()} for i, label in zip(ids, preds)]

    logger.info(f"Saving predictions to {PATH_TO_OUTPUT}")
    save_output(preds, PATH_TO_OUTPUT)
    logger.info("====================")
    logger.info("Finished successfully.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--task", "-t", help="Path to a RSG dataset", required=True)
    arg("--elmo", "-e", help="Path to a forlder with ELMo model", required=True)
    arg(
        "--elmo_layers",
        help="What ELMo layers to use?",
        default="average",
        choices=["average", "all", "top"],
    )
    arg(
        "--pooling",
        help="Add a pooling layer on the full sequence or return the last output only",
        default=False,
        action='store_true'
    )
    arg(
        "--activation",
        "-a",
        help="softmax or sigmoid for a final activation function?",
        default="softmax",
        choices=["softmax", "sigmoid"],
    )
    arg(
        "--num_epochs",
        "-n",
        help="number of epochs to train a keras model",
        type=int,
        default=15,
    )
    arg(
        "--hidden_size",
        help="size of hidden layers",
        type=int,
        default=16,
    )
    arg(
        "--batch_size",
        help="batch size for elmo and keras",
        type=int,
        default=32,
    )

    args = parser.parse_args()
    PATH_TO_DATASET = args.task
    PATH_TO_ELMO = args.elmo
    ELMO_LAYERS = args.elmo_layers
    POOLING = args.pooling
    ACTIVATION = args.activation
    EPOCHS = args.num_epochs
    HIDDEN_SIZE = args.hidden_size
    BATCH_SIZE = args.batch_size

    TASK_NAME_FIRST_CHAR = re.search('[A-Z]+.*', PATH_TO_DATASET).span()[0]

    log_format = f"%(asctime)s : %(levelname)s : %(message)s"
    start_time = time.strftime('%d%m%Y_%H%M%S', time.localtime())
    logging.basicConfig(format=log_format,
                        filename="logs/%s_%s.log" % (
                            PATH_TO_DATASET[TASK_NAME_FIRST_CHAR:-1], start_time),
                        filemode="w", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # For reproducibility:
    np.random.seed(42)
    python_random.seed(42)
    # tf.random.set_seed(42)

    logger.info(f"Following parameters were used")
    logger.info(f"Task: {PATH_TO_DATASET}, elmo_model: {PATH_TO_ELMO}")
    logger.info(f"ELMO_LAYERS: {ELMO_LAYERS}")
    logger.info(f"Pooling: {POOLING}, Activation function: {ACTIVATION}")
    logger.info(
        f"Hidden_size: {HIDDEN_SIZE}, Batch_size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    logger.info(f"=======================")

    main(
        PATH_TO_DATASET, TASK_NAME_FIRST_CHAR,
        PATH_TO_ELMO, ELMO_LAYERS,
        POOLING, ACTIVATION, EPOCHS, 
        HIDDEN_SIZE, BATCH_SIZE)
