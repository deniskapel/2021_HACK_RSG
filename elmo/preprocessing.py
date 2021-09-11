from ufal.udpipe import Model, Pipeline
from dataset_utils.utils import save_output, TEXT_FIELDS
from tqdm import tqdm
import os
import sys
import argparse
import codecs
import json


UD_MODEL = Model.load("models/ud/russian-taiga-ud-2.5-191206.udpipe")
UD_PIPELINE = Pipeline(
    UD_MODEL, "tokenizer", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
    )


def main(use_lemmas: bool, input_dir: str):
    # print(use_lemmas, input_dir, output_dir)
    tasks = [task for task in os.listdir(input_dir) if task in TEXT_FIELDS]
    [preprocess_task(input_dir, t, use_lemmas) for t in tqdm(tasks)]


def preprocess_task(input_dir: str, task: str, use_lemmas: bool):
    """ replaces """
    output_dir = 'data/lemmatised/' if args.lemmas else 'data/tokenised/'
    
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

        # out = [preprocess_line(line, use_lemmas, task) for line in lines]

        out = []    
        for line in lines:
            out.append(preprocess_line(line, use_lemmas, task))

        save_output(out, output_dir + task + "/" + sample)
        
    
def preprocess_line(line: dict, use_lemmas: bool, task: str) -> dict:
    """ tokenise/lemmatize only necessary fields in each entry of the task """

    if task == 'MuSeRC':

        line['passage']['text'] = preprocess(line['passage']['text'], use_lemmas)
        # inner fields
        for i, question in enumerate(line['passage']['questions']):
            line['passage']['questions'][i]['question'] = preprocess(
                line['passage']['questions'][i]['question'], use_lemmas)
            for j, answer in enumerate(question['answers']):
                line['passage']['questions'][i][
                    'answers'][j]['text'] = preprocess(
                        line['passage']['questions'][i]['answers'][j]['text'],
                        use_lemmas)
        return line
    
    elif task == 'RuCoS':
        line['passage']['text'] = preprocess(
            line['passage']['text'], use_lemmas)
        # inner fields
        for i, entry in enumerate(line['qas']):
            line['qas'][i]['query'] = preprocess(
                line['qas'][i]['query'], use_lemmas)
        return line

    # preprocessor for other tasks with simple structure
    fields = TEXT_FIELDS[task]
    for field in fields:
        line[field] = preprocess(line[field], use_lemmas)

    return line


def preprocess(text: str, use_lemmas: bool) -> str:
    """ 
        lemmatize or tokenize sentence

        returns a string where there is a space character
        between each token, including punctuation.

        The UD model is hard-coded as the very same model was used 
        to preprocess training data, and no change is planned for
        the sake of experiment
    """
    processed = UD_PIPELINE.process(text)
    # skip unnecessary lines 
    content = [line for line in processed.split("\n") if not line.startswith("#")]

    # extract UD feature
    tagged = [w.split("\t") for w in content if w]
    
    output = []
    
    for t in tagged:
        # can be simplified to extract t[1] or t[2] only
        # but left if any additional feature are required in the future 
        if len(t) != 10:
            continue
        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
        
        output.append(token if not use_lemmas else lemma)


    output = " ".join(output)
    return output



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    # tokenise or lemmatise necessary collumns in each dataset    
    arg("--lemmas", help="Lemmatize?", default=False, action='store_true')
    arg("--input_dir", '-d', help="Where are the original datasets stored?", default='data/combined/')
    
    args = parser.parse_args()

    main(args.lemmas, args.input_dir)