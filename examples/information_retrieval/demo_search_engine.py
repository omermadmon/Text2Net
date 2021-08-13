import os
import networkx as nx
from collections import namedtuple
from Text2Net import Text2Net

EXPANDED_QUERY = 'liverpool vs manchester united premier league match'.split(' ')


def relevance_probability(document, query):
    """
    An estimator of the probability of the event 'document is relevant w.r.t query'
    Computed by summation of pagerank scores in the document of the token in the expanded query.
    :param document: pagerank scores of the network representing the document.
    :param query: token of expanded query (as list).
    :return: An estimator of the probability of relevance.
    """
    return sum([document[token] for token in set(document.keys()).intersection(set(query))])


if __name__ == '__main__':
    # Read documents
    documents = []
    documents_dict = {}
    Document = namedtuple("Document", "id text")
    for filename in os.listdir('data/documents'):
        if filename.endswith(".txt"):
            with open(f'data/documents/{filename}') as f:
                id = filename.split('.')[0]
                text = f.read()
                documents.append(Document(id=id, text=text))
                documents_dict[id] = documents[-1]

    # Create a PageRank-based inverted index:
    index = dict()
    for doc in documents:
        doc_as_graph = Text2Net(doc.text).transform(n_nodes=100, weight_function='jaccard')
        doc_page_rank = nx.pagerank(doc_as_graph, alpha=0.9)
        index[doc.id] = doc_page_rank

    # Rank documents by probability of relevance:
    ranked_docs = [(doc_id, relevance_probability(doc_pagerank, EXPANDED_QUERY))
                   for doc_id, doc_pagerank in index.items()]
    ranked_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)
    print(f'Query: {EXPANDED_QUERY}')
    for doc_id, score in ranked_docs:
        print(f'{doc_id}: {documents_dict[doc_id].text}')