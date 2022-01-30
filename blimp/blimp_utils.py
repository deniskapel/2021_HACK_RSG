import json

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

def load(filename):
    pairs = []
    with open(filename) as f:
        for line in f.readlines():
            pairs.append(json.loads(line))
    return pairs


def get_token_logp(token: dict, vocab_size=30_003) -> tuple:
    """ returns token logp from forward and backward lstm """
    vocab_forward = dict(zip(token['forward']['candidate_words'], token['forward']['logp']))
    vocab_backward = dict(zip(token['backward']['candidate_words'], token['backward']['logp']))
    forward_logp = vocab_forward.get(token['word'], vocab_forward['<UNK>'])
    backward_logp = vocab_backward.get(token['word'], vocab_backward['<UNK>'])

    return forward_logp, backward_logp


def get_ppl(sentence, direction='forward', vocab_size=30_003):

    err_message = "Direction must be either 'forward', 'backward' or 'bidirectional'"
    assert direction in ['forward', 'backward', 'bidirectional'], err_message
    log_p = [get_token_logp(token, vocab_size) for token in sentence]

    if direction == 'forward':
        log_p = [f for f, b in log_p]
    elif direction == 'backward':
        log_p = [b for f, b in log_p]
    else:
        log_p = [np.mean([f,b]) for f, b in log_p]

    ppl = np.sum(log_p)

    return ppl


def run(model, dataloader, direction):
    correct = 0

    vocab_size = 30_003

    for good, bad in dataloader:
        good = model.get_elmo_substitutes(good, topn=vocab_size)
        bad = model.get_elmo_substitutes(bad, topn=vocab_size)

        for good_sent, bad_sent in zip(good, bad):

            good_ppl = get_ppl(good_sent, direction, vocab_size=vocab_size)
            bad_ppl = get_ppl(bad_sent, direction, vocab_size=vocab_size)

            if good_ppl > bad_ppl:
                correct += 1

    return correct / len(dataloader.dataset)



class Blimp:
    def __init__(self):
        self.phenomena = {}

    def add_result(self, phenomenon, uid, accuracy):
        if phenomenon not in self.phenomena:
            self.phenomena[phenomenon] = {}
        self.phenomena[phenomenon][uid] = accuracy

    def __str__(self):
        def iterator():
            for phenomenon_key in sorted(self.phenomena.keys()):
                phenomenon = self.phenomena[phenomenon_key]
                for uid_key in sorted(phenomenon.keys()):
                    yield f"{phenomenon_key},{uid_key},{phenomenon[uid_key]}"
        return '\n'.join(iterator())



class BlimpDataset(Dataset):

    """ customized Dataset class from torch """

    def __init__(self, data: list, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """ shape each sample into a proper """
        pair = self.data[index]
        good = " ".join([token.text for token in self.tokenizer(pair["sentence_good"])])
        bad = " ".join([token.text for token in self.tokenizer(pair["sentence_bad"])])

        return good, bad

def collate_fn(batch) -> tuple:
    goods, bads = list(), list()

    for good, bad in batch:
        goods.append(good)
        bads.append(bad)

    return goods, bads
