import pandas as pd

if __name__ == '__main__':
    # Data preprocessing and transformation
    print('reading data. . .')
    artists = pd.read_csv('data/artists-data.csv')
    lyrics = pd.read_csv('data/lyrics-data.csv')

    # Create an artists-lyrics unified dataframe for each genre
    print('preprocessing data. . .')
    artists['Artist'] = artists['Artist'].apply(lambda x: x.lower())
    lyrics = lyrics.loc[lyrics['Idiom'] == 'ENGLISH']
    lyrics['Artist'] = lyrics['ALink'].apply(lambda x: x[1:-1].replace('-', ' '))

    top_k = 50
    df_dict = {x['Genre'].values[0]: x for _, x in artists.groupby(artists['Genre'])}
    df_dict = {k: v for k, v in df_dict.items() if k in ['Hip Hop', 'Pop', 'Rock']}
    df_dict = {k: v.sort_values(by=['Popularity', 'Songs'], ascending=[False, False])[:top_k] for k, v in
               df_dict.items()}

    for genre, df in df_dict.items():
        df_dict[genre] = df.merge(lyrics, on='Artist')[['Artist', 'Popularity', 'Genre', 'SName', 'Lyric']]
        df_dict[genre].to_csv(f'data/{genre}_data.csv')
