import sys
import random as python_random
import numpy as np
from collections import Counter
# from simple_elmo import ElmoModel
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import to_categorical
# from sklearn.metrics import classification_report
from dataset_utils.utils import RSG_MorphAnalyzer, keras_model, save_output
# from dataset_utils.muserc import get_MuSeRC_predictions, MuSeRC_metrics
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
    # val, _ = build_features('%sval.jsonl' % (PATH_TO_DATASET))
    X_train = morph.normalize_sentences(train[0], use_lemmas=USE_LEMMAS)
    print(train)


if __name__ == '__main__':
    # For reproducibility:
    np.random.seed(42)
    python_random.seed(42)
    # tf.random.set_seed(42)

    main(sys.argv[1:])
