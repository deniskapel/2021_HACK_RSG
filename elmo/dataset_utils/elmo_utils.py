from tensorflow import Graph
from simple_elmo import ElmoModel
from numpy import swapaxes

def load_elmo(path: str, batch_size: int, method=None):
    """
        loads elmo model and returns it with its graph
        to handle multiple graph issues.

        Both are used in extract_embeddings()
    """
    if method == "simple":
        elmo_model = ElmoModel()
        elmo_model.load(path, batch_size)
        elmo_graph = None
    else:
        graph = Graph()
        with graph.as_default() as elmo_graph:
            elmo_model = ElmoModel()
            elmo_model.load(path, batch_size)
    return elmo_model, elmo_graph


def reshape4Dto3D(emb):
    """
    Reshapes embeddings returned by simple elmo with elmo_layers='all'.
    In this case, the embeddings have 4 dimensions but LSTM input is 3D.
    The function concatenates features from all layers into one dimension
    1. swap axes of layers and sequences: a view is returned
    2. merge two last dimensions
    """
    return swapaxes(emb, 1, 2).reshape(emb.shape[0],emb.shape[2],-1)
