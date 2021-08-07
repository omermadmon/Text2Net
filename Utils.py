import string
import nltk
from nltk.corpus import wordnet
from nltk import tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import networkx as nx
import matplotlib.pyplot as plt


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


def jaccard_weight(semantic_units, word_i, word_j):
    # TODO: add documentation.
    counter = 0
    for su in semantic_units:
        if word_i in su and word_j in su:
            counter += 1
    return counter / len([su for su in semantic_units if word_i in su or word_j in su])


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
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
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


def semantic_units_to_graph(semantic_units, n_nodes):
    # TODO: add documentation.
    # TODO: turn weight into a argument and extract from the default function flow.
    words = [word for SU in semantic_units for word in SU]
    words_set = set(words)
    top_n_words = sorted(list(words_set), key=lambda x: -words.count(x))[:n_nodes]
    graph_dict = {}
    for i, word_i in enumerate(top_n_words):
        for j, word_j in enumerate(top_n_words):
            weight = jaccard_weight(semantic_units, word_i, word_j)
            if i < j and weight > 0:
                if word_i in graph_dict.keys() and word_j not in graph_dict[word_i].keys():
                    graph_dict[word_i][word_j] = {'weight': weight}
                else:
                    graph_dict[word_i] = {word_j: {'weight': weight}}
    return nx.from_dict_of_dicts(graph_dict)


def visualize(G, title=None, nodes_factor=1, edges_factor=1, node_color='#abe2ed', edgecolors='#42c2db'):
    weights = [edges_factor * G[u][v]['weight'] for u, v in G.edges()]
    d = dict(G.degree)
    nx.draw_networkx(G, node_color=node_color, edgecolors=edgecolors, width=weights,
                     node_size=[nodes_factor * v for v in d.values()])
    if title is not None:
        plt.title(title)
    plt.show()
