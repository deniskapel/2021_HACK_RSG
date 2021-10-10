import argparse
import logging
import sys
import re
import random as python_random

import numpy as np
from tensorflow.random import set_seed
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef

from dataset_utils.features import build_features
from dataset_utils.utils import save_output, DataGenerator
from dataset_utils.global_vars import TIMESTAMP
from dataset_utils.elmo_utils import load_elmo
from dataset_utils.keras_utils import (
    keras_model,
    early_stopping,
    wrap_checkpoint)


def main(
    path_to_task: str, task_name_first_char: int, path_to_elmo: str, 
    pooling: bool, shuffle: bool, activation: str,
    epochs: int, hidden_size: int, batch_size: int):

    TASK_NAME = path_to_task[task_name_first_char:-1]
    INPUT_FOLDER = path_to_task[:task_name_first_char]
    PATH_TO_OUTPUT = 'submissions/%s.jsonl' % (TASK_NAME)

    """ DATA """
    if TASK_NAME in ['MuSeRC', 'RuCoS']:
        sys.stderr.write(
            'Check README to see how to make submissions for these tasks\n')
        sys.exit(1)
    elif TASK_NAME == 'LiDiRus':
        train, _ = build_features('%sTERRa/train.jsonl' % (INPUT_FOLDER))
        val, _ = build_features('%sTERRa/val.jsonl' % (INPUT_FOLDER))
        test, ids = build_features('%sLiDiRus.jsonl' % (path_to_task))
    else:
        train, _ = build_features('%strain.jsonl' % (path_to_task))
        val, _ = build_features('%sval.jsonl' % (path_to_task))
        test, ids = build_features('%stest.jsonl' % (path_to_task))

    logger.info(f"=======================")
    logger.info(f"loading Elmo model")
    elmo_model, elmo_graph = load_elmo(path_to_elmo, batch_size)

    # extract samples from sample+label bundles
    X_train = list(zip(*train[0]))
    X_valid = list(zip(*val[0]))

    # tokenize each sample in a set
    X_train = [[sample.split() for sample in part] for part in X_train]
    X_valid = [[sample.split() for sample in part] for part in X_valid]
    # get max_length for each sample part, 
    # e.g. 7 for premise and 5 for hypothesis 
    max_lengths = [np.max([len(sample) for sample in part]) for part in X_train]
    # reshape labels
    classes = sorted(list(set(train[1])))
    y_train = [classes.index(i) for i in train[1]]
    num_classes = len(classes)
    y_valid = [classes.index(i) for i in val[1]]

    # parameters for a DataGenerator instance 
    params = {
        'max_lengths': max_lengths,
        'batch_size': batch_size,
        'n_classes': num_classes,
        'elmo_model': elmo_model,
        "elmo_graph": elmo_graph}

    training_generator = DataGenerator(
        X_train, y_train, shuffle=shuffle, **params)
    validation_generator = DataGenerator(
        X_valid, y_valid, shuffle=False, **params)

    # Warm up elmo as it works better when first applied to dummy data  
    training_generator[0]

    """ MODEL """
    # initialize a keras model that takes elmo embeddings as its input
    model = keras_model(n_features=elmo_model.vector_size,
                        MAXLEN=sum(max_lengths),
                        hidden_size=hidden_size,
                        num_classes=num_classes,
                        pooling=pooling,
                        activation=activation)

    model.summary(print_fn=logger.info)

    logger.info(f"Start training")
    logger.info("====================")

    model.fit(
        x = training_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[
            wrap_checkpoint(f'{TASK_NAME}_{TIMESTAMP}'),
            early_stopping])


    """ PREDICTING """
    logger.info("====================")
    logger.info("Start predicting.")

    # get score on a validation dataset
    preds = model.predict(validation_generator)
    preds = [classes[int(np.argmax(pred))] for pred in preds]
    if TASK_NAME == 'LiDiRus':
        logger.info(f"MC score is {matthews_corrcoef(val[1], preds)}")
    else:
        logger.info(classification_report(val[1], preds))

    """ APPLY TO A TEST SET """
    # Preprocess the test split
    X_test = list(zip(*test[0]))
    X_test = [[sample.split() for sample in part] for part in X_test]
    # test[1] stores None values, replace with default value 
    y_test = [0 for i in test[1]]
    test_generator = DataGenerator(
        X_test, y_test, shuffle=False, **params)
    # generate predictiions
    preds = model.predict(test_generator)
    preds = [classes[int(np.argmax(pred))] for pred in np.around(preds)]
    preds = [
        {"idx": i, "label": str(l).lower()} for i, l in zip(ids, preds)
        ]
    # save in the RSG format
    logger.info(f"Saving predictions to {PATH_TO_OUTPUT}")
    save_output(preds, PATH_TO_OUTPUT)
    logger.info("====================")
    logger.info("Finished successfully.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--task", "-t", 
        help="Path to a RSG dataset folder, e.g. data/tokenised/TERRa/", 
        required=True)
    arg("--elmo", "-e", help="Path to a folder with ELMo model", required=True)
    arg(
        "--pooling",
        help="Add a pooling layer on the full sequence or return the last output only",
        default=False,
        action='store_true'
    )
    arg(
        "--shuffle",
        help="Add shuffle on each epoch end?",
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
        default=128,
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
    SHUFFLE = args.shuffle
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
    set_seed(42)

    logger.info(f"Following parameters were used")
    logger.info(f"Task: {PATH_TO_DATASET}, elmo_model: {PATH_TO_ELMO}")
    logger.info(f"Pooling: {POOLING}, Activation function: {ACTIVATION}")
    logger.info(f"Shuffle on each epoch end: {SHUFFLE}")
    logger.info(
        f"Hidden_size: {HIDDEN_SIZE}, Batch_size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    logger.info(f"=======================")

    main(
        PATH_TO_DATASET, TASK_NAME_FIRST_CHAR,
        PATH_TO_ELMO, POOLING, SHUFFLE, ACTIVATION,
        EPOCHS, HIDDEN_SIZE, BATCH_SIZE)
