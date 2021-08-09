def indicator_weight_function(word_i, word_j, semantic_units):
    # TODO: add documentation.
    for su in semantic_units:
        if word_i in su and word_j in su:
            return 1
    return 0


def jaccard_weight_function(word_i, word_j, semantic_units):
    # TODO: add documentation.
    counter = 0
    for su in semantic_units:
        if word_i in su and word_j in su:
            counter += 1
    return counter / len([su for su in semantic_units if word_i in su or word_j in su])


weight_functions = {
    'indicator': indicator_weight_function,
    'jaccard': jaccard_weight_function
}