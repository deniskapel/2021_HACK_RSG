"""
based on
    https://github.com/RussianNLP/RussianSuperGLUE/blob/master/tfidf_baseline
"""
import codecs
import json
import sys


def build_features(path):
    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))
    # datasets are different, did not bother with generalizing
    if 'RWSD' in path:
        res = list(map(build_feature_RWSD, lines))
    elif 'PARus' in path:
        res = list(map(build_feature_PARus, lines))
    elif 'MuSeRC' in path:
        res = list(map(build_feature_MuSeRC, lines))
    elif 'RUSSE' in path:
        res = list(map(build_feature_RUSSE, lines))
    elif 'TERRa' in path:
        res = list(map(build_feature_TERRa, lines))
    elif 'LiDiRus' in path:  # need to update the dataset folder
        res = list(map(build_feature_LiDiRus, lines))
    elif 'RCB' in path:
        res = list(map(build_feature_RCB, lines))
    elif 'RuCoS' in path:
        res = list(map(build_feature_RuCoS, lines))
    elif 'DaNetQA' in path:
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


def build_feature_RuCoS(row):
    # TODO: Does not work for now
    psg = row["passage"]["text"].replace("\n@highlight\n", " ")

    # extract entities from text as strings
    ent_idxs = row["passage"]["entities"]
    ents = [row["passage"]["text"][idx["start"]: idx["end"] + 1]
            for idx in ent_idxs]

    qas = row["qas"]
    queries = []
    labels = []
    return labels


def build_feature_MuSeRC(row):
    # TODO: Does not work for now
    text = row["passage"]["text"]
    qa = {}
    labels = []
    
    for line in row["passage"]["questions"]:
        qa[line['question']] = []

        for answ in line["answers"]:
            labels.append(answ.get("label", 0))
            qa[line['question']].append(answ['text'])

    return (text, qa), labels
