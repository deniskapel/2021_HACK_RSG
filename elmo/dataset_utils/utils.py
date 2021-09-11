import json

def save_output(data, path):
    """ a function to properly save the output before its submission """
    with open(path, mode="w") as file:
        for line in sorted(data, key=lambda x: int(x.get("idx"))):
            line["idx"] = int(line["idx"])
            file.write(f"{json.dumps(line, ensure_ascii=False)}\n")

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
