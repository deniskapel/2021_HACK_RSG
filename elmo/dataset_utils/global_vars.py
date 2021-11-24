import time

from spacy.lang.ru import Russian
from spacy.lang.en import English

TIMESTAMP = time.strftime('%d%m%Y_%H%M%S', time.localtime())

DTYPE = 'float64'

TEXT_FIELDS = {
    "DaNetQA": ["question", 'passage'],
    'LiDiRus': ['sentence1', 'sentence2'],
    "MuSeRC": [], # hard coded due to a complex structure
    "PARus": ['premise', 'choice1', 'choice2'],
    'RCB': ['premise', 'hypothesis'],
    "RuCoS": [], # hard coded due to a complex structure
    'RUSSE': ['sentence1', 'sentence2'],
    'RWSD': ['text'],
    'TERRa': ['premise', 'hypothesis']
}

TOKENIZERS = {
    'Russian': Russian().tokenizer,
    'English': English().tokenizer}
