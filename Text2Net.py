from Utils import semantic_units_to_graph
from SemanticUnitsParsing import text_to_semantic_units_by_sentence
from nltk.corpus import stopwords, wordnet
from string import punctuation

DEFAULT_STOPWORDS = stopwords.words('english') + list(punctuation)
DEFAULT_PUNCTUATION = list(punctuation)
DEFAULT_POS_TO_INCLUDE = tuple([wordnet.NOUN, wordnet.VERB, wordnet.ADV, wordnet.ADJ])


class Text2Net:
    # TODO: add documentation.

    def __init__(self, text):
        self.text = text

    def transform(self, n_nodes, weight_function='indicator',
                  pos_to_include=DEFAULT_POS_TO_INCLUDE,
                  stop_words=DEFAULT_STOPWORDS + DEFAULT_PUNCTUATION):
        # TODO: add documentation.
        # TODO: add arguments to diffuse into inner functions (SU to graph, text to SU).
        semantic_units = text_to_semantic_units_by_sentence(self.text,
                                                            pos_to_include=pos_to_include,
                                                            stop_words=stop_words)
        return semantic_units_to_graph(semantic_units,
                                       n_nodes,
                                       weight_function=weight_function)
