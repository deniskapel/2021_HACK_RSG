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
        sys.stderr.write('dataset is not ready yet\n')
        sys.exit(1)
    elif 'RUSSE' in path:
        res = list(map(build_feature_RUSSE, lines))
    elif 'TERRa' in path:
        res = list(map(build_feature_TERRa, lines))
    elif 'LiDiRus' in path:  # need to update the dataset folder
        res = list(map(build_feature_LiDiRus, lines))
    elif 'RCB' in path:
        res = list(map(build_feature_RCB, lines))
    elif 'RuCoS' in path:
        sys.stderr.write('dataset is not ready yet\n')
        sys.exit(1)
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
        hypothesis = row["hypothesis"]
    else:
        premise = str(row["sentence1"]).strip()
        hypothesis = row["sentence2"]
    label = row.get("label")
    res = f"{premise} {hypothesis}"
    return res, label


def build_feature_DaNetQA(row):
    res = str(row["question"]).strip()
    res = str(row["passage"]).strip()
    label = row.get("label")
    return res, label


def build_feature_RWSD(row):
    premise = str(row["text"]).strip()
    span1 = row["target"]["span1_text"]
    span2 = row["target"]["span2_text"]
    label = row.get("label")
    res = f"{premise} {span1} {span2}"
    return res, label


def build_feature_PARus(row):
    premise = str(row["premise"]).strip()
    choice1 = row["choice1"]
    choice2 = row["choice2"]
    label = row.get("label")
    question = "Что было ПРИЧИНОЙ этого?" if row["question"] == "cause" else "Что случилось в РЕЗУЛЬТАТЕ?"
    res = f"{premise} {question} {choice1} {choice2}"
    return res, label


def build_feature_RUSSE(row):
    sentence1 = row["sentence1"].strip()
    sentence2 = row["sentence2"].strip()
    word = row["word"].strip()
    label = row.get("label")
    res = f"{sentence1} {sentence2} {word}"
    return res, label


def build_feature_TERRa(row):
    premise = str(row["premise"]).strip()
    hypothesis = row["hypothesis"]
    label = row.get("label")
    res = f"{premise} {hypothesis}"
    return res, label


def build_feature_RCB(row):
    premise = str(row["premise"]).strip()
    hypothesis = row["hypothesis"]
    label = row.get("label")
    res = f"{premise} {hypothesis}"
    return res, label


def build_feature_RuCoS(row):
    pass


def build_feature_MuSeRC(row):
    pass