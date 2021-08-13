from Utils import semantic_units_to_graph
from SemanticUnitsParsing import text_to_semantic_units_by_sentence
from nltk.corpus import stopwords, wordnet
from string import punctuation

DEFAULT_STOPWORDS = stopwords.words('english') + list(punctuation)
DEFAULT_PUNCTUATION = list(punctuation)
DEFAULT_POS_TO_INCLUDE = tuple([wordnet.NOUN, wordnet.VERB, wordnet.ADV, wordnet.ADJ])


class Text2Net:
    """
    A class designated for tranforming text into a network.
    The output network is an undirected, possibly weighted, nx graph.
    """

    def __init__(self, text):
        self.text = text.lower()

    def transform(self, n_nodes, weight_function='indicator',
                  pos_to_include=DEFAULT_POS_TO_INCLUDE,
                  stop_words=DEFAULT_STOPWORDS + DEFAULT_PUNCTUATION):
        """
        Output a graph representation of self.text.
        :param n_nodes: number of nodes (most frequent tokens) to be included in the output graph.
        :param weight_function: weight function of the graph edges (by default - unweighted graph).
        :param pos_to_include: part-of-speech tags to include.
        :param stop_words: list of words to exclude from the output semantic units.
        :return: A nx.Graph object.
        """
        semantic_units = text_to_semantic_units_by_sentence(self.text,
                                                            pos_to_include=pos_to_include,
                                                            stop_words=stop_words)
        return semantic_units_to_graph(semantic_units,
                                       n_nodes,
                                       weight_function=weight_function)
