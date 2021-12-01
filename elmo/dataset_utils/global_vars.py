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
    'TERRa': ['premise', 'hypothesis'],
    'AX-b': ['sentence1', 'sentence2'],
    'AX-g': ['hypothesis', 'premise'],
    'BoolQ': ["question", 'passage'],
    'CB': ['premise', 'hypothesis'],
    'COPA': ['premise', 'choice1', 'choice2'],
    'MultiRC': [], # hard coded due to a complex structure
    'ReCoRD': [], # hard coded due to a complex structure
    'RTE': ['premise', 'hypothesis'],
    'WiC': ['sentence1', 'sentence2'],
    'WSC': ['text']
}
