import functools
import codecs
import json

import numpy as np

from dataset_utils.global_vars import DTYPE
from dataset_utils.elmo_utils import reshape4Dto3D


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


def get_row_pred_MuSeRC(
    row: dict, elmo_model, elmo_layers, n_features, elmo_session,
    keras_model, max_lengths: list):
    """
        returns properly shaped predictions and true lables per row.
        The third output is a dict to upload predictions to the leaderboard.
    """
    dim2 = sum(max_lengths)
    # put text entries into a list to extract embeddings properly
    text = [row["passage"]["text"].split()]
    text = elmo_model.get_elmo_vectors(
        text, warmup=False, layers=elmo_layers, session=elmo_session)

    if elmo_layers == 'all':
        text = reshape4Dto3D(text)

    # truncate if sequence len is longer than maxlen
    text = text[:,:max_lengths[0],:]

    res = []
    labels = []
    res_ids = {"idx": row["idx"], "passage": {"questions": []}}

    for line in row["passage"]["questions"]:
        # store all the answers per question
        res_line = {"idx": line["idx"], "answers": []}
        line_answers = []
        line_labels = []

        question = [line["question"].split()]
        question = elmo_model.get_elmo_vectors(
            question, warmup=False, layers=elmo_layers, session=elmo_session)

        for answ in line["answers"]:
            line_answers.append(answ['text'].split())
            line_labels.append(answ.get("label", 0))

        # extract embeddings from all the answers
        line_answers = elmo_model.get_elmo_vectors(
            line_answers, warmup=False,
            layers=elmo_layers, session=elmo_session)

        # create dummy array to store embeddings
        emb = np.zeros(
            (line_answers.shape[0], dim2, n_features),
            dtype=DTYPE)

        if elmo_layers == 'all':
            question =  reshape4Dto3D(question)
            line_answers =  reshape4Dto3D(line_answers)

        # truncate if sequence len is longer than maxlen
        question = question[:,:max_lengths[1],:]
        line_answers = line_answers[:,:max_lengths[2],:]

        # store a text in every sample
        emb[:, :text.shape[1], :] = text
        # store a question in every sample after a text
        emb[:, max_lengths[0]:max_lengths[0] + question.shape[1], :] = question
        # store all the answers right after the text and question
        starting_idx = max_lengths[0]+max_lengths[1]
        emb[:,starting_idx:starting_idx + line_answers.shape[1],:] = line_answers

        # some rows may include > 32 samples,
        # so model.predict(x) and not model(x) is used
        preds = keras_model.predict(emb)
        preds = [int(np.argmax(pred)) for pred in preds]
        res.append(preds)
        labels.append(line_labels)

        for answ, p in zip(line["answers"], preds):
            res_line["answers"].append({"idx": answ["idx"], "label": p})
        res_ids["passage"]["questions"].append(res_line)
    return res, labels, res_ids


def get_MuSeRC_predictions(
    path: str, elmo_model, elmo_layers, n_features,
    elmo_session, keras_model, max_lengths: list):
    """ a function to get predictions in a MuSeRC order """
    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))

    preds = []
    labels = []
    res = []

    for row in lines:
        pred, lbls, res_ids = get_row_pred_MuSeRC(
            row, elmo_model, elmo_layers, n_features, elmo_session,
            keras_model, max_lengths)
        preds.extend(pred)
        labels.extend(lbls)
        res.append(res_ids)

    return preds, labels, res



def tokenize_muserc(dataset: list) -> list:
    """
        tokenizes each part of the dataset
        dataset[0] - 1D list of passages
        dataset[1] - 2D list of questions
        dataset[2] - 3D list of answers
    """
    passages = [sample.split() for sample in dataset[0]]
    questions = [[q.split() for q in qs] for qs in dataset[1]]
    answers = [
        [[a.split() for a in anss] for anss in answs] for answs in dataset[2]]

    return passages, questions, answers


def align_passage_question_answer(data: list) -> list:
    """
        reshapes features for training:
        [p1,p2], [[p1_q1, p1_q2], [p2_q1]],
        [[[p1_q1_a1, p1_q1_a2], [p1_q2_a1, p1_q2_a2]], [[p2_q1_a1, p2_q1_a2]]]

        ->
        [p1,p1,p1,p1,p2,p2], [p1_q1, p1_q1, p1_q2, p1_q2, p2_q1, p2_q1],
        [p1_q1_a1, p1_q1_a2, p1_q2_a1, p1_q2_a2, p2_q1_a1, p2_q1_a2]

        passages and questions are multiplied by the number of answers
        This ensure triplets (passage, question, answer)
    """
    output = [[],[],[]]
    var1 = 0
    var2 = 0
    # align passage with all its questions and answers
    for passage, questions, answers_p in zip(data[0], data[1], data[2]):
        # align question with all its answers
        for question, answers_q in zip(questions, answers_p):
            for answer in answers_q:
                output[0].append(passage)
                output[1].append(question)
                output[2].append(answer)

    return output
