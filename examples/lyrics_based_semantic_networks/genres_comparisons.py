import pickle
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations, product
from scipy.stats import ttest_ind
from collections import defaultdict

if __name__ == '__main__':
    with open('data/measures_dict.pickle', 'rb') as handle:
        measures_dict = pickle.load(handle)

    # Descriptive statistics - plotting measures distributions per genre
    print('plotting measures distributions per genre. . .')
    genres = ['Rock', 'Hip Hop', 'Pop']
    measures = ['ASPL', 'AVG_DEG', 'CC']

    for measure in measures:
        df = pd.DataFrame()
        for genre, d in measures_dict.items():
            df[genre] = pd.Series(d[measure])
        boxplot = df.boxplot()
        boxplot.set_title(f'{measure} Distributions Per Genre')
        plt.savefig(f'figures/{measure}_distributions_per_genre')
        plt.close()

    # Multiple comparisons - all pairwise hypothesis
    g_couples = list(combinations(genres, r=2))
    hypothesis = list(product(g_couples, measures))
    pv_dict = {}
    for genres, measure in hypothesis:
        T, pv = ttest_ind(measures_dict[genres[0]][measure], measures_dict[genres[1]][measure], equal_var=False)
        pv_dict[(genres, measure)] = {'T': T, 'P-value': pv, 'Bonf. Adj. P-value': len(hypothesis)*pv,
                                      'Rejected': len(hypothesis)*pv < 0.05}

    pvalues_dataframe_rows = {k: defaultdict(int) for k in g_couples}
    for key, value in pv_dict.items():
        ((genre1, genre2), measure) = (key[0][0], key[0][1]), key[1]
        pv = value['Bonf. Adj. P-value']
        pvalues_dataframe_rows[(genre1, genre2)][measure] = pv

    results = pd.DataFrame(pvalues_dataframe_rows.values(),
                           index=[f'{genres[0]} - {genres[1]}' for genres in pvalues_dataframe_rows.keys()])
    results.to_csv('results.csv')