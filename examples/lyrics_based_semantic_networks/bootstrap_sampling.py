import pickle
import random
import pandas as pd
import networkx as nx
from Text2Net import Text2Net
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import ttest_ind
from itertools import combinations
from collections import defaultdict

if __name__ == '__main__':
    df_dict = dict()
    for genre in ['Rock', 'Pop', 'Hip Hop']:
        df_dict[genre] = pd.read_csv(f'data/{genre}_data.csv')

    # Bootstrap sampling - create num_samples networks for each genre,
    # each network made of sample_size songs' lyrics
    print('start bootstrap sampling. . .')
    measures_dict = {
        'Hip Hop': {'ASPL': [], 'CC': [], 'AVG_DEG': []},
        'Pop': {'ASPL': [], 'CC': [], 'AVG_DEG': []},
        'Rock': {'ASPL': [], 'CC': [], 'AVG_DEG': []}
    }

    sample_size = 100
    num_samples = 50

    for genre in ['Rock', 'Hip Hop', 'Pop']:
        i = 0
        while i < num_samples:
            lyrics = '\n' \
                .join([df_dict[genre]['Lyric'][i] for i in random.sample(range(0, len(df_dict[genre]['Lyric'])),
                                                                         sample_size)])
            G = Text2Net(lyrics).transform(n_nodes=100, weight_function='jaccard')
            try:
                measures_dict[genre]['ASPL'].append(nx.average_shortest_path_length(G))
                measures_dict[genre]['CC'].append(nx.algorithms.cluster.average_clustering(G))
                degs = [pair[1] for pair in G.degree]
                measures_dict[genre]['AVG_DEG'].append(sum(degs) / len(degs))
                i += 1
            except nx.NetworkXError:
                print('Generated network is not fully connected.')
            print(f'    Genre: {genre}, Sample number: {i} was sampled.')
    print('finished bootstrap sampling. . .')

    print('dumping pickle. . .')
    with open('data/measures_dict.pickle', 'wb') as handle:
        pickle.dump(measures_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)