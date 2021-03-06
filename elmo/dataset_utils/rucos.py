from collections import Counter
import string
import re
import codecs
import json

import numpy as np

from dataset_utils.global_vars import DTYPE
from dataset_utils.elmo_utils import reshape4Dto3D


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = [0]
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset: list, predictions):
    f1 = exact_match = total = 0
    correct_ids = []
    for prediction, passage in zip(predictions, dataset):
        prediction = prediction["label"]
        for qa in passage['qas']:
            total += 1
            ground_truths = list(map(lambda x: x['text'], qa.get("answers", "")))

            _exact_match = metric_max_over_ground_truths(exact_match_score, prediction,
                                                         ground_truths)
            if int(_exact_match) == 1:
                correct_ids.append(qa['idx'])
            exact_match += _exact_match

            f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = exact_match / total
    f1 = f1 / total
    return exact_match, f1


def get_rucos_predictions(
        path: str, elmo_model, elmo_layers, n_features,
        elmo_session, keras_model, max_lengths: list, offset=0):
    """ a function to get predictions in a RuCoS order """
    filename = path[re.search('(val)|(test).jsonl', path).span()[0]:]

    if 'RuCoS' in path:
        path_to_raw_file = f'data/combined/RuCoS/{filename}'
    else:
        path_to_raw_file = f'data/eng/ReCoRD/{filename}'
        offset = 1

    with codecs.open(path_to_raw_file, encoding='utf-8-sig') as reader:
        """
            Entities are encoded with indices.
            After preprocessing, indices shift.
            To extract entities, original files are needed.
        """
        raw_lines = reader.read().split("\n")
        raw_lines = list(map(json.loads, filter(None, raw_lines)))

    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))

    preds = []

    for row, raw_row in zip(lines, raw_lines):
        pred = get_row_pred(
            row, raw_row, elmo_model, elmo_layers,
            n_features, elmo_session, keras_model,
            max_lengths, offset=offset)
        preds.append({
            "idx": row["idx"],
            "label": pred
        })
    return lines, preds


def get_row_pred(
        row: dict, raw_row: dict,
        elmo_model, elmo_layers, n_features, elmo_session, keras_model,
        max_lengths: list, offset: int = 0):
    text = [row["passage"]["text"].replace("@highlight", " ").split()]
    text = elmo_model.get_elmo_vectors(
        text, warmup=False, layers=elmo_layers, session=elmo_session)

    if elmo_layers == 'all':
        text = reshape4Dto3D(text)

    # truncate if sequence len is longer than maxlen
    text = text[:,:max_lengths[0],:]

    words = [
        raw_row["passage"]["text"][x["start"]: x["end"]+offset]
        for x in raw_row["passage"]["entities"]]

    # create dummy array to store embeddings
    emb = np.zeros(
        (len(words), sum(max_lengths), n_features), dtype=DTYPE)

    # store text in every sample
    emb[:, :text.shape[1], :] = text

    res = []

    for line in row["qas"]:
        queries = []
        for word in words:
            queries.append(line["query"].replace("@placeholder", word).split())

        queries = elmo_model.get_elmo_vectors(
            queries, warmup=False, layers=elmo_layers, session=elmo_session)

        if elmo_layers == 'all':
            queries = reshape4Dto3D(queries)

        # truncate if sequence len is longer than maxlen
        queries = queries[:,:max_lengths[1],:]
        # store queries right after texts, ~ hstack
        emb[:, max_lengths[0]:max_lengths[0]+queries.shape[1], :] = queries

        preds = keras_model.predict(emb)
        # choose a prediction with a largest prob of being true
        pred_idx = preds[:, 1].argsort()[-1]
        # transform an id to an actual prediction
        pred = np.array(words)[pred_idx]
        res.append(pred)

    return " ".join(res)


def tokenize_rucos(dataset: list, cut=None) -> list:
    passages = [sample.split()[:cut] for sample in dataset[0]]
    queries = [[q.split() for q in qs] for qs in dataset[1]]

    return passages, queries


def align_passage_queries(data: tuple) -> list:
    """
    reshapes features for training creating copies
    of text part

    ([p1,p2],[[q1,q2],[q3,q4]]) ->
    [[p1, p1, p2, p2], [q1, q2, q3, q4]]
    """
    output = [[], []]

    for passage, queries in zip(data[0], data[1]):
        for query in queries:
            # aling passage and query
            output[0].append(passage)
            output[1].append(query)

    return output
