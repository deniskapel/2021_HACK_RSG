import os
import sys
import argparse
import codecs
import json

from tqdm import tqdm

from dataset_utils.utils import save_output
from dataset_utils.global_vars import TEXT_FIELDS, TOKENIZERS


def main(input_dir: str, language: str):
    tasks = [task for task in os.listdir(input_dir) if task in TEXT_FIELDS]
    [preprocess_task(input_dir, t, TOKENIZERS[language]) for t in tqdm(tasks)]

def preprocess_task(input_dir: str, task: str, preproc_fn):
    """ replaces raw texts with preprocessed ones """
    output_dir = 'data/tokenised/'
    if not os.path.isdir(output_dir + task):
        # create directories for preprocessed tasks
        os.makedirs(output_dir + task)

    samples = [s for s in os.listdir(input_dir+task) if not s.startswith('.')]

    for sample in samples:
        source_file = input_dir + task + '/' + sample
        out = []
        with codecs.open(source_file, encoding='utf-8-sig') as r:
            lines = r.read().split("\n")
            lines = list(map(json.loads, filter(None, lines)))

        out = [preprocess_line(line, task, preproc_fn) for line in lines]

        save_output(out, output_dir + task + "/" + sample)


def preprocess_line(line: dict, task: str, preproc_fn) -> dict:
    """ preprocess only necessary fields in each entry of the task """

    if task == 'MuSeRC':
        line['passage']['text'] = preprocess(line['passage']['text'], preproc_fn)
        # inner fields
        for i, question in enumerate(line['passage']['questions']):
            line['passage']['questions'][i]['question'] = preprocess(
                line['passage']['questions'][i]['question'], preproc_fn)
            for j, answer in enumerate(question['answers']):
                line['passage']['questions'][i]['answers'][j]['text'] = preprocess(
                    line['passage']['questions'][i]['answers'][j]['text'], preproc_fn)
        return line

    elif task == 'RuCoS':
        line['passage']['text'] = preprocess(
            line['passage']['text'], preproc_fn)
        # inner fields
        for i, entry in enumerate(line['qas']):
            line['qas'][i]['query'] = preprocess(
                line['qas'][i]['query'], preproc_fn)
        return line

    # preprocessor for other tasks with simple structure
    fields = TEXT_FIELDS[task]
    for field in fields:
        line[field] = preprocess(line[field], preproc_fn)

    return line


def preprocess(text: str, preproc_fn) -> str:
    """
        lemmatize or tokenize sentence

        returns a string where there is a space character
        between each token, including punctuation.

        The UD model is hard-coded as the very same model was used
        to preprocess training data, and no change is planned for
        the sake of experiment
    """
    return " ".join([token.text for token in preproc_fn(text)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input_dir", '-d', help="Where are the original datasets stored?", default='data/combined/')
    arg("--language", '-l', help="Input language", default='Russian', choices=["Russian", "English"])
    args = parser.parse_args()
    main(args.input_dir, args.language)
