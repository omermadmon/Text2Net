from nltk import tokenize, pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from Utils import tag_pos


def text_to_semantic_units_by_sentence(text, pos_to_include, stop_words, lemmatizer=WordNetLemmatizer()):
    """
    Parse the given text into semantic units by sentences in the text.
    :param text: a string text to be broken into semantic units.
    :param pos_to_include: part-of-speech tags to include.
    :param stop_words: list of words to exclude from the output semantic units.
    :param lemmatizer: a component for assigning base forms to tokens.
    :return: A list of semantic units (as lists).
    """
    semantic_units = []
    for sentence in tokenize.sent_tokenize(text):
        pos_tagged = pos_tag(word_tokenize(sentence))
        wordnet_tagged = list(map(lambda x: (x[0], tag_pos(x[1])), pos_tagged))
        lemmatized_semantic_units = []
        for word, tag in wordnet_tagged:
            if word not in stop_words and tag in pos_to_include:
                lemmatized_semantic_units.append(lemmatizer.lemmatize(word, tag))
            elif tag is None and word not in stop_words:
                lemmatized_semantic_units.append(word)
        if len(lemmatized_semantic_units) > 0:
            semantic_units.append(lemmatized_semantic_units)
    return semantic_units
