from nltk import tokenize, pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from Utils import tag_pos


def text_to_semantic_units_by_sentence(raw_text,
                                       pos_to_include=[wordnet.NOUN, wordnet.VERB, wordnet.ADV, wordnet.ADJ],
                                       stop_words=stopwords.words('english'),
                                       lemmatizer=WordNetLemmatizer()):
    # TODO: add documentation.
    text = raw_text.lower()
    semantic_units = []
    for sentence in tokenize.sent_tokenize(text):
        # TODO: decide about next line:
        # sentence = sentence.replace("'", "")
        pos_tagged = pos_tag(word_tokenize(sentence))
        wordnet_tagged = list(map(lambda x: (x[0], tag_pos(x[1])), pos_tagged))
        lemmatized_semantic_units = []
        for word, tag in wordnet_tagged:
            if word not in stop_words and tag in pos_to_include:
                lemmatized_semantic_units.append(lemmatizer.lemmatize(word, tag))
            # TODO: remove redundant lines:
            # if word == '``' or (word != 'im' and len(word) <= 2):
            #     continue
            # elif tag is None and word not in string.punctuation and word not in stop_words:
            #     # if there is no available tag, append the token as is
            #     # lemmatized_SU.append(word)
            #     continue
            # elif word not in string.punctuation and word not in stop_words:
            #     # else use the tag to lemmatize the token
            #     if tag in (wordnet.NOUN):
            #         lemmatized_semantic_units.append(lemmatizer.lemmatize(word, tag))
        semantic_units.append(lemmatized_semantic_units)
    return semantic_units
