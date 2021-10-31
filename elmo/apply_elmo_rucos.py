import argparse
import logging
import sys
import re
import random as python_random
import numpy as np
from tensorflow.random import set_seed
from dataset_utils.features import build_features
from dataset_utils.utils import save_output, DataGenerator
from dataset_utils.elmo_utils import load_elmo
from dataset_utils.keras_utils import (
    keras_model,
    early_stopping,
    wrap_checkpoint)
from dataset_utils.global_vars import TIMESTAMP
from dataset_utils.rucos import (
    tokenize_rucos,
    align_passage_queries,
    evaluate,
    get_rucos_predictions)


def main(
        path_to_task: str, task_name_first_char: int, path_to_elmo: str,
        pooling: bool, shuffle: bool, activation: str,
        epochs: int, hidden_size: int, batch_size: int):
    task_name = path_to_task[task_name_first_char:-1]

    # Why do we need this variable at all?
    input_folder = path_to_task[:task_name_first_char]

    if task_name != 'RuCoS':
        sys.stderr.write(
            'Check README to see which file to run for this task\n')
        sys.exit(1)

    path_to_output = 'submissions/%s.jsonl' % task_name

    # tf.debugging.set_log_device_placement(True)

    logger.info(f"=======================")
    logger.info(f"loading Elmo model")
    # if method == "simple", the model will be loaded in the regular way
    elmo_model, elmo_graph = load_elmo(path_to_elmo, 32, method="graph")
    elmo_layers = "top"

    train, _ = build_features('%strain.jsonl' % path_to_task)
    val, _ = build_features('%sval.jsonl' % path_to_task)

    # extract samples from sample+label bundles
    x_train = list(zip(*train[0]))
    x_valid = list(zip(*val[0]))

    # tokenize each sample in a set
    x_train = tokenize_rucos(x_train)
    x_train = align_passage_queries(x_train)
    x_valid = tokenize_rucos(x_valid)
    x_valid = align_passage_queries(x_valid)

    # get max_length for each sample part,
    # e.g. 135 for passages and 32 for queries
    max_lengths = [np.max([len(sample) for sample in part]) for part in x_train]

    # reshape labels
    y_train = [sample for subset in train[1] for sample in subset]
    classes = sorted(list(set(y_train)))
    y_train = [classes.index(i) for i in y_train]
    num_classes = len(classes)
    y_valid = [sample for subset in val[1] for sample in subset]
    y_valid = [classes.index(i) for i in y_valid]

    # parameters for a DataGenerator instance
    params = {
        'max_lengths': max_lengths,
        'batch_size': batch_size,
        'n_classes': num_classes,
        'elmo_model': elmo_model,
        "elmo_graph": elmo_graph,
        "layers": elmo_layers}

    training_generator = DataGenerator(
        x_train, y_train, shuffle=shuffle, **params)
    validation_generator = DataGenerator(
        x_valid, y_valid, shuffle=False, **params)

    # Warm up elmo as it works better when first applied to dummy data
    _ = training_generator[0]

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
        training_generator,
        epochs=epochs,
        validation_data=validation_generator,
        verbose="auto",
        batch_size=batch_size,
        callbacks=[
            wrap_checkpoint(f'{task_name}_{TIMESTAMP}'),
            early_stopping])

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(f'checkpoints/{task_name}_{TIMESTAMP}/checkpoint')

    """ PREDICTING """
    logger.info("====================")
    logger.info("Start predicting.")
    """ Get validation score """

    # Prediction is done for each passage_query set separately
    infer_params = {
        'elmo_model': elmo_model, 'elmo_layers': elmo_layers,
        'elmo_session': training_generator.tf_session,  
        'keras_model': model, 'max_lengths': max_lengths}

    dataset, preds = get_rucos_predictions(
        '%sval.jsonl' % path_to_task, **infer_params)
    logger.info(f" EM, F1a scores on validation are {evaluate(dataset, preds)}")

    """ APPLY TO A TEST SET """
    _, test_preds = get_rucos_predictions(
        '%stest.jsonl' % path_to_task, **infer_params)

    logger.info(f"Saving predictions to {path_to_output}")
    save_output(test_preds, path_to_output)
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
    logging.basicConfig(format=log_format, level=logging.INFO)
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
