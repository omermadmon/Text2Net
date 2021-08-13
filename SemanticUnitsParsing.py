from nltk import tokenize, pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from Utils import tag_pos


def text_to_semantic_units_by_sentence(raw_text, pos_to_include, stop_words, lemmatizer=WordNetLemmatizer()):
    # TODO: add documentation.
    text = raw_text.lower()
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
