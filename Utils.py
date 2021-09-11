import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from WeightFunctions import weight_functions


# TODO: figure out how to make sure nltk requirements are downloaded:
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')


def tag_pos(nltk_tag):
    # TODO: add documentation.
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def semantic_units_to_graph(semantic_units, n_nodes, weight_function):
    """
    Given a list of lists of token representing the semantic units of a text, create a nx.Graph
    representation of the text.
    :param semantic_units: A list of semantic units (as lists).
    :param n_nodes: number of nodes (most frequent tokens) to be included in the output graph.
    :param weight_function: weight function of the graph edges.
    :return: A nx.Graph object.
    """
    if weight_function not in weight_functions.keys():
        raise ValueError(f'weight_function must be one of: {weight_functions.keys()}')
    words = [word for SU in semantic_units for word in SU]
    words_set = set(words)
    top_n_words = sorted(list(words_set), key=lambda x: -words.count(x))[:n_nodes]
    graph_dict = {}
    for i, word_i in enumerate(top_n_words):
        for j, word_j in enumerate(top_n_words):
            weight = weight_functions[weight_function](word_i, word_j, semantic_units)
            if i < j and weight > 0:
                if word_i in graph_dict.keys() and word_j not in graph_dict[word_i].keys():
                    graph_dict[word_i][word_j] = {'weight': weight}
                else:
                    graph_dict[word_i] = {word_j: {'weight': weight}}
    return nx.from_dict_of_dicts(graph_dict)


def visualize(G, title=None, nodes_factor=1, edges_factor=1, node_color='#abe2ed',
              edgecolors='#42c2db', figsize=(15, 10), save_fig=None):
    """
    Visualizing network.
    :param G: a network to visualize.
    :param title: title of the plot.
    :param nodes_factor: a factor to determine node size in visualization.
    :param edges_factor: a factor to determine edge width in visualization.
    :param node_color: node color in visualization.
    :param edgecolors: edge color in visualization.
    :param figsize: tuple representing figure size.
    :param save_fig: a path to save the figure. If not provided figure is shown via plt.show().
    :return: None.
    """
    plt.figure(figsize=figsize)
    weights = [edges_factor * G[u][v]['weight'] for u, v in G.edges()]
    d = dict(G.degree)
    nx.draw_networkx(G, node_color=node_color, edgecolors=edgecolors, width=weights,
                     node_size=[nodes_factor * v for v in d.values()])
    if title is not None:
        plt.title(title)
    if save_fig:
        plt.savefig(save_fig)
        plt.close()
    else:
        plt.show()
