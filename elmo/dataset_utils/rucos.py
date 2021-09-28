import jsonlines
from collections import Counter
import string
import re
import sys
from itertools import chain

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.extmath import softmax as sk_softmax


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

            _exact_match = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
            if int(_exact_match) == 1:
                correct_ids.append(qa['idx'])
            exact_match += _exact_match

            f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = exact_match / total
    f1 = f1 / total
    return exact_match, f1


def get_RuCoS_predictions(
    path: str, elmo_model, keras_model,
    max_lengths: list, dtype: str):
    """ a function to get predictions in a RuCoS order """
    filename = path[re.search('(val)|(test).jsonl', path).span()[0]:]
    path_to_raw_file = f'data/combined/RuCoS/{filename}'
    with jsonlines.open(path_to_raw_file) as reader:
        """
            Entities are encoded with indices. 
            After preprocessing, indices shift.
            To extract entities, original files are needed.
        """
        raw_lines = list(reader)
    
    with jsonlines.open(path) as reader:
        lines = list(reader)
    
    preds = []
    
    for row, raw_row in zip(lines, raw_lines):
        pred = get_row_pred(
            row, raw_row,
            elmo_model, keras_model,
            max_lengths, dtype)
        preds.append({
            "idx": row["idx"],
            "label": pred
        })
    return lines, preds


def get_row_pred(
    row:dict, raw_row:dict, 
    elmo_model, keras_model,
    max_lengths: list, dtype: str):

    text = [row["passage"]["text"].replace("@highlight", " ").split()]
    text = elmo_model.get_elmo_vectors(text)
    text = pad_sequences(
        text, maxlen=max_lengths[0],
        dtype=dtype, padding='post')

    res = []
    words = [
        raw_row["passage"]["text"][x["start"]: x["end"]]
        for x in raw_row["passage"]["entities"]]

    for line in row["qas"]:
        queries = []
        for word in words:
            queries.append(
                line["query"].replace("@placeholder", word).split()
                )
        
        queries = elmo_model.get_elmo_vectors(queries)
        queries = pad_sequences(
            queries, maxlen=max_lengths[1],
            dtype=dtype, padding='post')

        sample = []        
        
        for i in range(queries.shape[0]):
            sample.append(
                np.hstack(
                    (text, np.array([queries[i]]))
            ))
        logits = keras_model.predict(np.vstack(sample))
        # BERT solution applies softmax to logits first
        logits = sk_softmax(logits)
        # get the id of a word with a highest probability of being correct
        pred_idx = logits[:, 1].argsort()[-1]
        # transform an id to an actual prediction        
        pred = np.array(words)[pred_idx]
        res.append(pred)

    return " ".join(res)

def tokenize_rucos(dataset: list) -> list:
    passages = [sample.split() for sample in dataset[0]]
    queries = [[q.split() for q in qs] for qs in dataset[1]]
            
    return passages, queries

def reshape_rucos(dataset: tuple) -> list:
    """
        reshape complex rucos structure to the structure
        used for other datasets, i.e. list of 2D lists
    """

    passages = dataset[0]
    queries = list(chain(*dataset[1]))
    
    return [passages, queries]

def get_rucos_shape(queries: list) -> list:
    """
        extract number of queries per passage of the rucos dataset

        [p1: [q1,q2,q3], p2: [q1,q2,q3,q4],
        p3: [q1, q2]] -> [3,4,2]
    """
    return [len(qs) for qs in queries]


def align_passage_queries(
    dataset: list,
    original_shape: list
    ) -> np.ndarray:
    """
        To avoid extracting embeddings multiple times,
        passages and queries were processed separately.

        this function reshapes embeddings like this 
        based on their original shape:
        (
            [p1,p2,p3..pn], [q1,q2, q3, q4..qn]
        )  ->     
        [[p1, q1], [p1, q2], [p1, q3], [p2, q4],
        [p2, q5], [p3, q6], [pn, q(n-1)], [pn, qn]]
    """
    data = []
    query_id = 0
    for i, num_queries in enumerate(original_shape):
        passage = dataset[0][i] # i is a passage_id
        last_query_id = query_id + num_queries
        queries = dataset[1][query_id:last_query_id]
        
        for query in queries:
            # merge passage question and anwer
            data.append(
                # add a complete sample to the dataset
                np.vstack((passage, query)))

            # reassign starting position
        query_id = last_query_id

    # transform a list of numpy arrays to a 3D array
    return np.array(data)