import json

import numpy as np

from scipy.special import log_softmax

from torch.utils.data import Dataset


def load(filename):
    pairs = []
    with open(filename) as f:
        for line in f.readlines():
            pairs.append(json.loads(line))
    return pairs


def get_token_logp(token: dict, softmax:bool=True) -> tuple:
    """ returns token logp from forward and backward lstm """
    forward_logits = token['forward']['logp']
    backward_logits = token['backward']['logp']
    
    if softmax:
        forward_logits = log_softmax(forward_logits)
        backward_logits = log_softmax(backward_logits)
    
    vocab_forward = dict(zip(token['forward']['candidate_words'], forward_logits))
    vocab_backward = dict(zip(token['backward']['candidate_words'], backward_logits))
    
    forward_logp = vocab_forward.get(token['word'], vocab_forward['<UNK>'])
    backward_logp = vocab_backward.get(token['word'], vocab_backward['<UNK>'])

    word = token['word'] if forward_logp != vocab_forward['<UNK>'] else '<UNK>'

    return forward_logp, backward_logp, word


def get_ppl(sentence):
    log_p = [get_token_logp(token) for token in sentence]

    log_p_forw = list()
    log_p_bidir = list()

    for f, b, t in log_p:
        log_p_forw.append(f)
        log_p_bidir.append(np.mean([f,b]))
    
    ppl_forw = np.sum(log_p_forw)
    ppl_bidir = np.sum(log_p_bidir)
    
    return ppl_forw, ppl_bidir, log_p


def run(model, dataloader):
    correct_forw = 0
    correct_bidir = 0
    vocab_size = model.vocab.size
    
    preds = list()

    for good, bad in dataloader:
        good_text = good
        bad_text = bad
        good = model.get_elmo_substitutes(good, topn=vocab_size)
        bad = model.get_elmo_substitutes(bad, topn=vocab_size)

        for good_sent, bad_sent in zip(good, bad):

            good_ppl_forw, good_ppl_bidir, good_preds = get_ppl(good_sent)
            good_sent = " ".join([token['word'] for token in good_sent])

            bad_ppl_forw, bad_ppl_bidir, bad_preds = get_ppl(bad_sent)
            bad_sent = " ".join([token['word'] for token in bad_sent])
            
            preds.append({
                'good_sentence': good_text,
                'good': good_preds, 
                'bad_sentence': bad_text,
                'bad': bad_preds})

            if good_ppl_forw > bad_ppl_forw:
                correct_forw += 1

            if good_ppl_bidir > bad_ppl_bidir:
                correct_bidir += 1


    correct_forw = correct_forw / len(dataloader.dataset)
    correct_bidir = correct_bidir / len(dataloader.dataset)
    
    
    return correct_forw, correct_bidir, preds


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
