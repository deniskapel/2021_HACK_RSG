import argparse
import glob
import logging
import json

import simple_elmo
from spacy.lang.en import English
from torch.utils.data import DataLoader
from tqdm import tqdm

from blimp_utils import (
    load, run, Blimp, BlimpDataset, collate_fn
)

BATCH_SIZE = 1


def main(elmo_path: str, blimp_path: str):
    elmo_model = simple_elmo.ElmoModel()
    elmo_model.load(elmo_path, max_batch_size=BATCH_SIZE, full=True)

    blimp_forw = Blimp()
    blimp_bidir = Blimp()
    tokenizer = English().tokenizer

    for dataset in tqdm(glob.glob(f'{blimp_path}*.jsonl')):
        logger.warning(dataset)
        dataset = load(dataset)
        loader = DataLoader(
            BlimpDataset(dataset, tokenizer),
            batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn)

        acc_forw, acc_bidir, dataset_preds = run(elmo_model, loader)
        blimp_forw.add_result(
            dataset[0]["linguistics_term"], dataset[0]["UID"], acc_forw)
        blimp_bidir.add_result(
            dataset[0]["linguistics_term"], dataset[0]["UID"], acc_bidir)

        filename = f'{dataset[0]["linguistics_term"]}___{dataset[0]["UID"]}'

        with open(f'preds/{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_preds, f, ensure_ascii=False, indent=2)
        

    f = open('scores_forward.txt', 'w')
    f.write(blimp_forw.__str__())
    f.close()
    b = open('scores_bidirectional.txt', 'w')
    b.write(blimp_bidir.__str__())
    b.close()

    logger.warning(blimp_forw)
    logger.warning(blimp_bidir)



if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.WARNING
    )
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--elmo", "-e", help="Path to a folder with ELMo model", required=True)
    arg(
        "--blimp", "-b",
        help="Path to a folder with blimp datatests", default='blimp/data/')

    args = parser.parse_args()

    main(args.elmo, args.blimp)
