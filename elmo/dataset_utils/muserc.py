import functools
import numpy as np
import codecs
import json
from itertools import chain

from tensorflow.keras.preprocessing.sequence import pad_sequences


class MuSeRCMetrics:

    @staticmethod
    def per_question_metrics(dataset, output_map):
        P = []
        R = []
        for n, example in enumerate(dataset):
            predictedAns = example
            correctAns = output_map[n]
            predictCount = sum(predictedAns)
            correctCount = sum(correctAns)
            assert math.ceil(sum(predictedAns)) == sum(
                predictedAns), "sum of the scores: " + str(sum(predictedAns))
            agreementCount = sum(
                [a * b for (a, b) in zip(correctAns, predictedAns)])
            p1 = (1.0 * agreementCount /
                  predictCount) if predictCount > 0.0 else 1.0
            r1 = (1.0 * agreementCount /
                  correctCount) if correctCount > 0.0 else 1.0
            P.append(p1)
            R.append(r1)

        pAvg = Measures.avg(P)
        rAvg = Measures.avg(R)
        f1Avg = 2 * Measures.avg(R) * Measures.avg(P) / \
            (Measures.avg(P) + Measures.avg(R))
        return [pAvg, rAvg, f1Avg]

    @staticmethod
    def exact_match_metrics_origin(dataset, output_map, delta):
        EM = []
        for n, example in enumerate(dataset):
            predictedAns = example
            correctAns = output_map[n]

            em = 1.0 if sum(
                [abs(i - j) for i, j in zip(correctAns, predictedAns)]) <= delta else 0.0
            EM.append(em)
        return Measures.avg(EM)

    @staticmethod
    def exact_match_simple(dataset, output_map):
        EM = []
        for n, example in enumerate(dataset):
            predictedAns = example
            correctAns = output_map[n]
            if predictedAns == correctAns:
                em = 1
            else:
                em = 0
            EM.append(em)
        return sum(EM)/len(EM)

    @staticmethod
    def per_dataset_metric(dataset, output_map):
        """
        dataset = [[0,1,1], [0,1]]
        output_map = [[0,1,0], [0,1]]
        """
        agreementCount = 0
        correctCount = 0
        predictCount = 0
        for n, example in enumerate(dataset):
            predictedAns = example
            correctAns = output_map[n]
            predictCount += sum(predictedAns)
            correctCount += sum(correctAns)
            agreementCount += sum([a * b for (a, b)
                                   in zip(correctAns, predictedAns)])

        p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
        r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
        return [p1, r1, 2 * r1 * p1 / (p1 + r1)]

    @staticmethod
    def avg(l):
        return functools.reduce(lambda x, y: x + y, l) / len(l)


def MuSeRC_metrics(pred, labels):
    metrics = MuSeRCMetrics()
    em = metrics.exact_match_simple(pred, labels)
    em0 = metrics.exact_match_metrics_origin(pred, labels, 0)
    f1 = metrics.per_dataset_metric(pred, labels)
    f1a = f1[-1]
    return em0, f1a


Measures = MuSeRCMetrics


def get_row_pred_MuSeRC(row: dict, maxlen: int,
                        elmo_model, keras_model, morph_model,
                        use_lemmas=True):
    """
        returns properly shaped predictions and true lables per row.
        The third output is a dict to upload predictions to the leaderboard.

        if use_lemmas=True, apply lemmatisation, otherwise tokenisation only
    """
    text = row["passage"]["text"]
    res = []
    labels = []
    res_ids = {"idx": row["idx"], "passage": {"questions": []}}
    for line in row["passage"]["questions"]:
        res_line = {"idx": line["idx"], "answers": []}
        batch = []
        line_labels = []
        for answ in line["answers"]:
            # get labels if there are any
            line_labels.append(answ.get("label", 0))
            # transform a line to preprocess extract embeddings
            answ = f"{text} {line['question']} {answ['text']}"
            batch.append(answ)

        batch = morph_model.normalize_sentences(batch, use_lemmas)
        # extract for one questions and all answers to it
        embeddings = elmo_model.get_elmo_vectors(batch)
        # shape val based on train dimensions
        embeddings = pad_sequences(embeddings, maxlen=maxlen)
        preds = keras_model.predict(embeddings)
        # map predictions to the binary {0, 1} range and
        # extract the id of the largest one
        preds = [int(np.argmax(pred)) for pred in np.around(preds)]
        res.append(preds)
        labels.append(line_labels)

        for answ, p in zip(line["answers"], preds):
            res_line["answers"].append({"idx": answ["idx"], "label": p})
        res_ids["passage"]["questions"].append(res_line)

    return res, labels, res_ids


def get_MuSeRC_predictions(path, max_len: int,
                           elmo_model, keras_model,
                           morph_model, use_lemmas=True):
    """ a function to get predictions in a MuSeRC order """
    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))

    preds = []
    labels = []
    res = []

    for row in lines:
        pred, lbls, res_ids = get_row_pred_MuSeRC(
            row, max_len, elmo_model, keras_model, morph_model, use_lemmas)
        preds.extend(pred)
        labels.extend(lbls)
        res.append(res_ids)

    return preds, labels, res



def tokenize_muserc(dataset: list) -> list:
    """
        shapes multiple choice datasets to avoid 
        extracting embeddings from same elements several times
    """
    passages = [sample.split() for sample in dataset[0]]
    questions = [[q.split() for q in qa.keys()] for qa in dataset[1]]
    answers = [
        [[ans.split() for ans in a] for a in qa.values()] for qa in dataset[1]]
            
    return passages, questions, answers


def get_muserc_shape(answers: list) -> list:
    """
        extract shapes of the muserc dataset

        [p1: [q1: [a1,a2,a3], q2: [a1,a2,a3,a4]],
        p2: [q1: [a1,a2]]] -> [[3, 4], [2]]

    """
    # extract number of answers per each question of a questions passage
    return [[len(answer) for answer in a] for a in answers]


def reshape_muserc(dataset: tuple) -> list:
    """
        reshape complex muserc structure to the structure
        used for other datasets, i.e. list of 2D lists
    """

    passages = dataset[0]
    answers = list(chain(*dataset[1])) # flatten 2D list
    questions = list(chain(*[qa for p in dataset[2] for qa in p]))
    
    return [passages, answers, questions]


def align_passage_question_answer(
    dataset: list,
    original_shape: list
    ) -> np.ndarray:
    """
        To avoid extracting embeddings multiple times,
        passages, questions and answers were processed separately.

        this function reshapes a dataset into the following form:
        (
            [p1,p2,p3], [[q1,q2], [q1, q2]],
            [[[a1,a2], [a1]], [[a1,a2], [a1,a2,a3]]]]
        )  ->     
        [[p1, q1, a1], [p1, q1, a2], [p1, q2, a1], [p2, q1, a1],
        [p2, q1, a2], [p2, q2, a1], [p2, q2, a2], [p2, q2, a3]]
    """
    data = []
    answer_id = 0
    for i, questions in enumerate(original_shape):
        passage = dataset[0][i] # i is a passage_id
        for j, num_answers in enumerate(questions):
            question = dataset[1][j] # j is a question_id in a qa set
            # shift idx
            last_answer_id = answer_id + num_answers
            # extract necessary answers
            answers = dataset[2][answer_id:last_answer_id]
            
            for answer in answers:
                # merge passage question and anwer
                data.append(
                    # add a complete sample to the dataset
                    np.vstack(
                        (passage, question, answer))
                        )

            # reassign starting position
            answer_id = last_answer_id

    # transform a list of numpy arrays to a 3D array
    return np.array(data)
