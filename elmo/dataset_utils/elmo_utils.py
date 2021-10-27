from tensorflow import Graph
from simple_elmo import ElmoModel


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

