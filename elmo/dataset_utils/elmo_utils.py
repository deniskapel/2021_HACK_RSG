from tensorflow import Graph
from simple_elmo import ElmoModel

def load_elmo(path: str, batch_size: int):
    """ 
        loads elmo model and returns it with its graph
        to handle multiple graph issues.

        Both are used in extract_embeddings()
    """
    graph = Graph()
    
    with graph.as_default() as elmo_graph:
        elmo_model = ElmoModel()
        elmo_model.load(path, batch_size)
    return elmo_model, elmo_graph

def extract_embeddings(elmo_model, elmo_graph, data: list):
    """ 
        extract embeddings handling multiple graph issue 
    params: 
        elmo_model, elmo_graph - returned by load_elmo()
        data - a list of lists of tokens    
        
    returns: 
        embeddings shaped (n_samples, max_length, elmo_model.vector_size)
    """
    with elmo_graph.as_default():
        # embeddings will be extracted by batches
        X = elmo_model.get_elmo_vectors(data, warmup=False)

    return X