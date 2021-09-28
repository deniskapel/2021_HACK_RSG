import argparse
import logging
import sys
import re
import random as python_random

import numpy as np
from simple_elmo import ElmoModel
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

from dataset_utils.features import build_features
from dataset_utils.utils import save_output
from dataset_utils.keras_utils import (
    keras_model,
    early_stopping,
    wrap_checkpoint)
from dataset_utils.muserc import (
    tokenize_muserc,
    get_muserc_shape,
    reshape_muserc,
    align_passage_question_answer,
    get_MuSeRC_predictions,
    MuSeRC_metrics)
from dataset_utils.global_vars import TIMESTAMP



def main(
    path_to_task: str, task_name_first_char: int, path_to_elmo: str, 
    pooling: bool, activation: str, epochs: int, hidden_size: int,
    batch_size: int):

    TASK_NAME = path_to_task[task_name_first_char:-1]
    INPUT_FOLDER = path_to_task[:task_name_first_char]

    if TASK_NAME != 'MuSeRC':
        sys.stderr.write(
            'Use apply_elmo.py or apply_elmo_rucos for this task\n')
        sys.exit(1)

    PATH_TO_OUTPUT = 'submissions/%s.jsonl' % (TASK_NAME)

    train, _ = build_features('%strain.jsonl' % (path_to_task))
    val, _ = build_features('%sval.jsonl' % (path_to_task))

    # extract samples from sample+label bundles
    X_train = list(zip(*train[0]))
    X_valid = list(zip(*val[0]))

    # tokenize each sample in a set
    X_train = tokenize_muserc(X_train)
    X_train_original_shape = get_muserc_shape(X_train[2])
    X_train = reshape_muserc(X_train)
    X_valid = tokenize_muserc(X_valid)
    X_valid_original_shape = get_muserc_shape(X_valid[2])
    X_valid = reshape_muserc(X_valid)

    # reshape dataset to reuse functions

    """ EMBEDDINGS """
    logger.info(f"=======================")
    logger.info(f"loading Elmo model")
    elmo = ElmoModel()
    elmo.load(path_to_elmo, max_batch_size=batch_size)

    # create train embeddings    
    X_train_embeddings = [elmo.get_elmo_vectors(part) for part in X_train]

    # get max_length for each sample part, 
    # e.g. 7 for premise and 5 for hypothesis 
    max_lengths = [part.shape[1] for part in X_train_embeddings]
    # Dtype for padding, otherwise rounded to int32

    X_train_embeddings = align_passage_question_answer(
        X_train_embeddings,
        X_train_original_shape)

    DTYPE = X_train_embeddings.dtype

    # create validate embeddings
    X_val_embeddings = [elmo.get_elmo_vectors(part) for part in X_valid]

    # add padding before each sentence using train maxlength
    X_val_embeddings = [pad_sequences(d, maxlen=l, dtype=DTYPE, padding='post')
                        for d, l in zip(X_val_embeddings, max_lengths)]


    X_val_embeddings = align_passage_question_answer(
        X_val_embeddings,
        X_valid_original_shape)

    del X_train, X_valid

    # reshape labels
    y_train = [sample for subset in train[1] for sample in subset]
    classes = sorted(list(set(y_train)))
    y_train = [classes.index(i) for i in y_train]
    num_classes = len(classes)
    y_train = to_categorical(y_train, num_classes)
    y_valid = [sample for subset in val[1] for sample in subset]
    y_valid = [classes.index(i) for i in y_valid]
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
        batch_size=batch_size,
        callbacks=[wrap_checkpoint(f'{TASK_NAME}_{TIMESTAMP}'), early_stopping])

    del X_train_embeddings, train

    """ PREDICTING """
    logger.info("====================")
    logger.info("Start predicting.")

    """ Get validation score """
    # Prediction is done for each passage_questions_answers set separately
    preds, y_true, _ = get_MuSeRC_predictions(
        '%sval.jsonl' % (path_to_task),
        elmo, model, 
        max_lengths, DTYPE)
    logger.info(f" em0, F1a scores on validation are {MuSeRC_metrics(preds, y_true)}")

    """ APPLY TO A TEST SET """
    _, _, test_preds = get_MuSeRC_predictions(
        '%stest.jsonl' % (path_to_task),
        elmo, model, 
        max_lengths, DTYPE)

    logger.info(f"Saving predictions to {PATH_TO_OUTPUT}")
    save_output(test_preds, PATH_TO_OUTPUT)
    logger.info("====================")
    logger.info("Finished successfully.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--task", "-t", help="Path to a RSG dataset", required=True)
    arg("--elmo", "-e", help="Path to a forlder with ELMo model", required=True)
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
    POOLING = args.pooling
    ACTIVATION = args.activation
    EPOCHS = args.num_epochs
    HIDDEN_SIZE = args.hidden_size
    BATCH_SIZE = args.batch_size

    TASK_NAME_FIRST_CHAR = re.search('[A-Z]+.*', PATH_TO_DATASET).span()[0]

    log_format = f"%(asctime)s : %(levelname)s : %(message)s"
    logging.basicConfig(format=log_format,
                        filename="logs/%s_%s.log" % (
                            PATH_TO_DATASET[TASK_NAME_FIRST_CHAR:-1], TIMESTAMP),
                        filemode="w", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # For reproducibility:
    np.random.seed(42)
    python_random.seed(42)
    tf.random.set_seed(42)

    logger.info(f"Following parameters were used")
    logger.info(f"Task: {PATH_TO_DATASET}, elmo_model: {PATH_TO_ELMO}")
    logger.info(f"Pooling: {POOLING}, Activation function: {ACTIVATION}")
    logger.info(
        f"Hidden_size: {HIDDEN_SIZE}, Batch_size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    logger.info(f"=======================")

    main(
        PATH_TO_DATASET, TASK_NAME_FIRST_CHAR,
        PATH_TO_ELMO, POOLING, 
        ACTIVATION, EPOCHS, 
        HIDDEN_SIZE, BATCH_SIZE)