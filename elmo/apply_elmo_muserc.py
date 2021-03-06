import argparse
import logging
import sys
import re
import random as python_random

import numpy as np
from tensorflow.random import set_seed
from sklearn.metrics import classification_report

from dataset_utils.features import build_features
from dataset_utils.utils import save_output, DataGenerator
from dataset_utils.elmo_utils import load_elmo
from dataset_utils.keras_utils import (
    keras_model,
    early_stopping,
    wrap_checkpoint)
from dataset_utils.muserc import (
    tokenize_muserc,
    align_passage_question_answer,
    get_MuSeRC_predictions,
    MuSeRC_metrics)
from dataset_utils.global_vars import TIMESTAMP



def main(
    path_to_task: str, task_name_first_char: int, path_to_elmo: str,
    elmo_layers: str, pooling: bool, shuffle: bool, activation: str,
    epochs: int, hidden_size: int, batch_size: int):

    TASK_NAME = path_to_task[task_name_first_char:-1]
    INPUT_FOLDER = path_to_task[:task_name_first_char]

    if TASK_NAME not in ['MuSeRC', 'MultiRC']:
        sys.stderr.write(
            'Use apply_elmo.py or apply_elmo_rucos for this task\n')
        sys.exit(1)

    PATH_TO_OUTPUT = f'submissions/{TASK_NAME}_{TIMESTAMP}.jsonl'

    logger.info(f"=======================")
    logger.info(f"loading Elmo model")
    # if method == "simple", the model will be loaded in the regular way
    elmo_model, elmo_graph = load_elmo(path_to_elmo, 16, method="graph")

    n_features = elmo_model.vector_size
    if elmo_layers == 'all':
        n_features *= elmo_model.n_layers

    train, _ = build_features('%strain.jsonl' % (path_to_task))
    val, _ = build_features('%sval.jsonl' % (path_to_task))
    # extract samples from sample+label bundles
    X_train = list(zip(*train[0]))
    X_valid = list(zip(*val[0]))
    # tokenize each sample in a set
    X_train = tokenize_muserc(X_train)
    X_train = align_passage_question_answer(X_train)
    X_valid = tokenize_muserc(X_valid)
    X_valid = align_passage_question_answer(X_valid)

    # get max_length for each sample part,
    # e.g. 37 for passages and 15 for questions and 3 for answers
    max_lengths = [np.max([len(sample) for sample in part]) for part in X_train]

    # get labels, p=passage, q=question, a=answer
    y_train = [a for p in train[1] for q in p for a in q]
    classes = sorted(list(set(y_train)))
    y_train = [classes.index(i) for i in y_train]
    num_classes = len(classes)
    y_valid = [a for p in val[1] for q in p for a in q]
    y_valid = [classes.index(i) for i in y_valid]

    # parameters for a DataGenerator instance
    params = {
        'max_lengths': max_lengths,
        'batch_size': batch_size,
        'n_classes': num_classes,
        'elmo_model': elmo_model,
        "elmo_graph": elmo_graph,
        "layers": elmo_layers,
        'n_features': n_features}

    training_generator = DataGenerator(
        X_train, y_train, shuffle=shuffle, **params)
    validation_generator = DataGenerator(
        X_valid, y_valid, shuffle=False, **params)

    # Warm up elmo as it works better when first applied to dummy data
    _ = training_generator[0]

    """ MODEL """
    # initialize a keras model that takes elmo embeddings as its input
    model = keras_model(n_features=n_features,
                        size_splits=max_lengths,
                        hidden_size=hidden_size,
                        num_classes=num_classes,
                        pooling=pooling,
                        activation=activation)

    model.summary(print_fn=logger.info)

    logger.info(f"Start training")
    logger.info("====================")
    model.fit(
        training_generator,
        epochs=epochs,
        validation_data=validation_generator,
        batch_size=batch_size,
        callbacks=[
            wrap_checkpoint(f'{TASK_NAME}_{TIMESTAMP}'),
            early_stopping])

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(f'checkpoints/{TASK_NAME}_{TIMESTAMP}/checkpoint')

    """ PREDICTING """
    logger.info("====================")
    logger.info("Start predicting.")

    """ Get validation score """
    # Prediction is done for each passage_questions_answers set separately
    infer_params = {
        'elmo_model': elmo_model, 'elmo_layers': elmo_layers,
        "n_features": n_features, 'elmo_session': training_generator.tf_session,
        'keras_model': model, 'max_lengths': max_lengths}

    preds, y_true, _ = get_MuSeRC_predictions(
        '%sval.jsonl' % (path_to_task), **infer_params)

    logger.info(f" em0, F1a scores on validation are {MuSeRC_metrics(preds, y_true)}")
    """ APPLY TO A TEST SET """
    _, _, test_preds = get_MuSeRC_predictions(
        '%stest.jsonl' % (path_to_task), **infer_params)

    logger.info(f"Saving predictions to {PATH_TO_OUTPUT}")
    save_output(test_preds, PATH_TO_OUTPUT)
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
        default=256,
    )
    arg(
        "--batch_size",
        help="batch size for elmo and keras",
        type=int,
        default=64,
    )
    arg(
        "--elmo_layers",
        "-l",
        help="What ELMo layers to use?",
        default="average",
        choices=["average", "all", "top"],
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
    LAYERS = args.elmo_layers
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
    logger.info(f"Elmo layers: {LAYERS}, Shuffle on each epoch end: {SHUFFLE}")
    logger.info(
        f"Hidden_size: {HIDDEN_SIZE}, Batch_size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    logger.info(f"=======================")

    main(
        PATH_TO_DATASET, TASK_NAME_FIRST_CHAR,
        PATH_TO_ELMO, LAYERS, POOLING, SHUFFLE, ACTIVATION,
        EPOCHS, HIDDEN_SIZE, BATCH_SIZE)
