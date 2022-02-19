import sys
from simple_elmo import ElmoModel
from blimp_utils import get_ppl
from spacy.lang.en import English

model = ElmoModel()
model.load(sys.argv[1], max_batch_size=1, full=True)

tokenizer = English().tokenizer

pairs = [("Could that window ever shut?", "That window could ever shut."),
         ("could that window ever shut?", "that window could ever shut."),
         ("Could that window ever shut.", "That window could ever shut?"),
         ("Can it mean something?", "Piece does mean something."),
         ("Has the river ever frozen?", "The river has ever frozen."),
         ("Is it really good?", "It is really good?"),
         ("Is it really good.", "It is really good.")]

print("=========================")
for pair in pairs:
    good, bad = pair
    print(good)
    print(get_ppl(model.get_elmo_substitutes([" ".join([token.text for token in tokenizer(good)])],
                                             topn=model.vocab.size)[0], "bidirectional"))
    print(bad)
    print(get_ppl(model.get_elmo_substitutes([" ".join([token.text for token in tokenizer(bad)])],
                                             topn=model.vocab.size)[0], "bidirectional"))
    print("=========================")



