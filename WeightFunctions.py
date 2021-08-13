def indicator_weight_function(word_i, word_j, semantic_units):
    """
    Returns 1 if words_i and word_j appear in the same semantic unit, 0 otherwise.
    :param word_i: first token.
    :param word_j: second token.
    :param semantic_units: semantic units (as list of lists of tokens).
    :return: 1 if words_i and word_j appear in the same semantic unit, 0 otherwise.
    """
    for su in semantic_units:
        if word_i in su and word_j in su:
            return 1
    return 0


def jaccard_weight_function(word_i, word_j, semantic_units):
    """
    Returns the jaccard distance of two tokens in terms of semantic units occurrences.
    :param word_i: first token.
    :param word_j: second token.
    :param semantic_units: semantic units (as list of lists of tokens).
    :return: #semantic units containing both word_i and word_j /
    #semantic units containing either word_i or word_j
    """
    counter = 0
    for su in semantic_units:
        if word_i in su and word_j in su:
            counter += 1
    return counter / len([su for su in semantic_units if word_i in su or word_j in su])


weight_functions = {
    'indicator': indicator_weight_function,
    'jaccard': jaccard_weight_function
}