"""
based on
    https://github.com/RussianNLP/RussianSuperGLUE/blob/master/tfidf_baseline
"""
import codecs
import json
import sys
import re

def build_features(path):
    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))
    # datasets are different, did not bother with generalizing
    if 'RWSD' in path or 'WSC' in path:
        res = list(map(build_feature_RWSD, lines))
    elif 'PARus' in path:
        res = list(map(build_feature_PARus, lines))
    elif 'COPA' in path:
        res = list(map(build_feature_COPA, lines))
    elif 'MuSeRC' in path or 'MultiRC' in path:
        res = list(map(build_feature_MuSeRC, lines))
    elif 'RUSSE' in path or 'WiC' in path:
        res = list(map(build_feature_RUSSE, lines))
    elif 'TERRa' in path or 'RTE' in path:
        res = list(map(build_feature_TERRa, lines))
    elif 'LiDiRus' in path or 'AX-b' in path or 'AX-g' in path:
        res = list(map(build_feature_LiDiRus, lines))
    elif 'RCB' in path or 'CB' in path:
        res = list(map(build_feature_RCB, lines))
    elif 'RuCoS' in path:
        # get location of the raw file
        filename = path[re.search('(train)|(val).jsonl', path).span()[0]:]
        path_to_raw_file = f'data/combined/RuCoS/{filename}'
        with codecs.open(path_to_raw_file, encoding='utf-8-sig') as reader:
            """
                Entities are encoded with indices.
                After preprocessing, indices shift.
                To extract entities, original files are needed.
            """
            raw_lines = reader.read().split("\n")
            raw_lines = list(map(json.loads, filter(None, raw_lines)))
        res = list(map(build_feature_RuCoS, lines, raw_lines))
    elif 'ReCoRD' in path:
        # get location of the raw file
        filename = path[re.search('(train)|(val).jsonl', path).span()[0]:]
        path_to_raw_file = f'data/eng/ReCoRD/{filename}'
        with codecs.open(path_to_raw_file, encoding='utf-8-sig') as reader:
            """
                Entities are encoded with indices.
                After preprocessing, indices shift.
                To extract entities, original files are needed.
            """
            raw_lines = reader.read().split("\n")
            raw_lines = list(map(json.loads, filter(None, raw_lines)))
        res = list(map(build_feature_ReCoRD, lines, raw_lines))
    elif 'DaNetQA' in path or 'BoolQ' in path:
        res = list(map(build_feature_DaNetQA, lines))
    else:
        sys.stderr.write('Path is not valid.\n')
        sys.exit(1)

    texts = list(map(lambda x: x[0], res))
    labels = list(map(lambda x: x[1], res))
    ids = [x["idx"] for x in lines]
    return (texts, labels), ids


def build_feature_LiDiRus(row):
    if row.get("sentence1") is None:
        premise = str(row["premise"]).strip()
        hypothesis = str(row["hypothesis"]).strip()
    else:
        premise = str(row["sentence1"]).strip()
        hypothesis = str(row["sentence2"]).strip()
    label = row.get("label")
    return (premise, hypothesis), label


def build_feature_DaNetQA(row):
    question = str(row["question"]).strip()
    passage = str(row["passage"]).strip()
    label = row.get("label")
    return (question, passage), label


def build_feature_RWSD(row):
    premise = str(row["text"]).strip()
    span1 = row["target"]["span1_text"]
    span2 = row["target"]["span2_text"]
    label = row.get("label")
    return (premise, span1, span2), label


def build_feature_PARus(row):
    premise = str(row["premise"]).strip()
    choice1 = row["choice1"]
    choice2 = row["choice2"]
    label = row.get("label")
    # if-else is taken from the tfidf baseline code.
    question = "Что было причиной этого ?" if row["question"] == "cause" else "Что произошло в результате ?"
    return (premise, question, choice1, choice2), label

def build_feature_COPA(row):
        premise = str(row["premise"]).strip()
        choice1 = row["choice1"]
        choice2 = row["choice2"]
        label = row.get("label")
        # if-else is taken from the tfidf baseline code.
        question = "What was the reason for that ?" if row["question"] == "cause" else "What was the result ?"
        return (premise, question, choice1, choice2), label

def build_feature_RUSSE(row):
    sentence1 = row["sentence1"].strip()
    sentence2 = row["sentence2"].strip()
    word = row["word"].strip()
    label = row.get("label")
    return (sentence1, sentence2, word), label


def build_feature_TERRa(row):
    premise = str(row["premise"]).strip()
    hypothesis = row["hypothesis"]
    label = row.get("label")
    return (premise, hypothesis), label


def build_feature_RCB(row):
    premise = str(row["premise"]).strip()
    hypothesis = row["hypothesis"]
    label = row.get("label")
    return (premise, hypothesis), label


def build_feature_RuCoS(row, raw_row):
    """
        Raw files to extract entities properly as
        preprocessing shifts indices
    """
    # check dataset_utils/rucos.py file for RuCoS function
    text = row["passage"]["text"].replace("@highlight", " ")
    labels = []
    queries = []

    for entry in row["passage"]["entities"]:
        # extract entities from a text as strings
        entity = raw_row["passage"]["text"][entry["start"]: entry["end"]]
        entry['text'] = entity
        label = 1 if entry in row['qas'][0]['answers'] else 0
        labels.append(label)

        queries.append(
            row["qas"][0]['query'].replace("@placeholder", entity)
            )

    return (text, queries), labels

def build_feature_ReCoRD(row, raw_row):
    """
        Raw files to extract entities properly as
        preprocessing shifts indices
    """
    # check dataset_utils/rucos.py file for RuCoS function
    text = row["passage"]["text"].replace("@highlight", " ")
    labels = []
    queries = []

    for entry in row["passage"]["entities"]:
        # extract entities from a text as strings
        entity = raw_row["passage"]["text"][entry["start"]: entry["end"]+1]
        entry['text'] = entity
        label = 1 if entry in row['qas'][0]['answers'] else 0
        labels.append(label)

        queries.append(
            row["qas"][0]['query'].replace("@placeholder", entity)
            )

    return (text, queries), labels


def build_feature_MuSeRC(row):
    text = row["passage"]["text"]
    labels = []
    questions = []
    answers = []

    for line in row["passage"]["questions"]:
        questions.append(line['question'])
        line_answers = []
        line_labels = []
        for answ in line["answers"]:
            line_labels.append(answ.get("label", 0))
            line_answers.append(answ['text'])

        answers.append(line_answers)
        labels.append(line_labels)

    return (text, questions, answers), labels
