import tensorflow as tf
from simple_elmo import ElmoModel

def load_elmo(path: str, batch_size: int):
    """ 
        loads elmo model and returns it with its graph
        to handle multiple graph issues.

        This one will be used to extract embeddings only
    """
    graph = tf.Graph()
    
    with graph.as_default() as elmo_graph:
        elmo_model = ElmoModel()
        elmo_model.load(path, batch_size)
    return elmo_model, elmo_graph

def extract_embeddings(elmo_model, elmo_graph, data: list):
    """ 
        extract embeddings handling multiple graph issue 
    params: 
        elmo_model, elmo_graph - returned by load_elmo function
        data - a list of lists of tokens    
        
    returns: 
        embeddings shaped (n_samples, max_length, elmo_model.vector_size)
    """
    with elmo_graph.as_default():
        X = elmo_model.get_elmo_vectors(data)

    return X