import sys
from collections import Counter
from utils import RSG_MorphAnalyzer, build_model
from simple_elmo import ElmoModel
import tensorflow as tf
from dataset_utils.features import build_features


def main(args):
    if len(args) != 2:
        sys.stderr.write(
            'Usage: test_elmo.py <path_to_dataset_folder> <path_to_model_zip>\n')
        sys.exit(1)

    PATH_TO_DATASET = args[0]
    PATH_TO_ELMO = args[1]

    morph = RSG_MorphAnalyzer()

    train, _ = build_features('%strain.jsonl' % (PATH_TO_DATASET))
    val, _ = build_features('%sval.jsonl' % (PATH_TO_DATASET))
    test, ids = build_features('%stest.jsonl' % (PATH_TO_DATASET))

    X_train = morph.lemmatize_sentences(train[0])
    y_train = train[1]
    X_valid = morph.lemmatize_sentences(val[0])
    y_valid = val[1]

    N_LABELS = len(set(y_train))

    # get embeddings
    elmo = ElmoModel()
    elmo.load(PATH_TO_ELMO)
    X_train_embeddings = elmo.get_elmo_vectors(X_train[0:10])
    dims = X_train_embeddings.shape
    MAX_LEN = dims[1]
    VOCAB_SIZE = dims[2]
    X_valid_embeddings = elmo.get_elmo_vectors(X_valid[0:10])

    # model
    model = build_model(MAX_LEN, VOCAB_SIZE, N_LABELS, PATH_TO_ELMO)
    print(model.summary())

    model.fit(X_train_embeddings, y_train,
              validation_data=[X_valid_embeddings, y_valid],
              batch_size=20000,
              epochs=10)

    # preds = model_predict(X_valid).reshape(-1)
    # print(preds)


if __name__ == '__main__':
    main(sys.argv[1:])
