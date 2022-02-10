import argparse
import glob
import logging

import simple_elmo
from spacy.lang.en import English
from torch.utils.data import DataLoader
from tqdm import tqdm

from blimp_utils import (
    load, run, Blimp, BlimpDataset, collate_fn
)

BATCH_SIZE = 1


def main(elmo_path: str, blimp_path: str, direction: "forward"):
    elmo_model = simple_elmo.ElmoModel()
    elmo_model.load(elmo_path, max_batch_size=BATCH_SIZE, full=True)

    blimp = Blimp()
    tokenizer = English().tokenizer

    for dataset in tqdm(glob.glob(f'{blimp_path}*.jsonl')):
        logger.warning(dataset)
        dataset = load(dataset)
        loader = DataLoader(
            BlimpDataset(dataset, tokenizer),
            batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn)

        accuracy = run(elmo_model, loader, direction=direction)
        blimp.add_result(dataset[0]["linguistics_term"], dataset[0]["UID"], accuracy)

    s = open('scores.txt', 'a')
    s.write(blimp.__str__())
    s.close()
    logger.warning(blimp)


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
    arg("--direction", "-d", help="LSTM forward, backward or both", default='forward')

    args = parser.parse_args()

    main(args.elmo, args.blimp, args.direction)
