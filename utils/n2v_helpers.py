import gensim
from gensim.models import KeyedVectors
from node2vec import Node2Vec
from typing import Type
import numpy as np
from nx_helpers import node_to_str
import networkx as nx
import rdflib


def load_nodevectors_model(n2v_model_path: str) -> Type[KeyedVectors]:
    return KeyedVectors.load(n2v_model_path)


def get_nodevectors_vector(
    nodevectors_model: Type[KeyedVectors], node: Type[nx.Graph.node]
) -> np.array:
    node_str = None
    if type(node) is not rdflib.term.Literal:
        node_str = node_to_str(node)
    else:
        node_str = str(node)
    vector = nodevectors_model.wv[node_str]
    return vector


def load_n2v_model(n2v_model_path: str) -> Type[gensim.models.Word2Vec]:
    return gensim.models.Word2Vec.load(n2v_model_path)


def get_n2v_vector(
    n2v_model: Type[gensim.models.Word2Vec], node: Type[nx.Graph.node]
) -> np.array:
    node_str = node_to_str(node)
    return n2v_model.wv.get_vector(node_str)
