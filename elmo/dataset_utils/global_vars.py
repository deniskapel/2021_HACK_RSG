import time

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

PAD_PARAMS = {'dtype': DTYPE, 'padding': 'post'}
