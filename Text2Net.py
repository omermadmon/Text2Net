from Utils import semantic_units_to_graph, text_to_semantic_units_by_sentence


class Text2Net:
    # TODO: add documentation.

    def __init__(self, text):
        self.text = text

    def transform(self, n_nodes, weight_function):
        # TODO: add documentation.
        # TODO: add arguments to diffuse into inner functions (SU to graph, text to SU).
        semantic_units = text_to_semantic_units_by_sentence(self.text)
        return semantic_units_to_graph(semantic_units, n_nodes, weight_function=weight_function)
