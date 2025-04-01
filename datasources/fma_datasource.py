import os
import json
import pandas as pd
from .datasource import Datasource

class FMADatasource(Datasource):

    def __init__(self, datasets_path):
        super().__init__(datasets_path,'fma')
        self.dataset_path = os.path.join(self.datasets_path,'fma','data')
        self.metadata_path = os.path.join(self.dataset_path,'fma_metadata')
        self.files_path = '/VIU/09MIAR/datasets/fma/fma_large'

        genres_df = pd.read_csv(os.path.join(self.metadata_path,'genres.csv'))
        genres_df = genres_df[['genre_id','title']]
        genres_df['title'] = genres_df['title'].str.lower()
        genres_df = genres_df.where(genres_df['title'].isin(self.GENRE_TITLE)).dropna()
        genres_df['genre_id'] = genres_df['genre_id'].astype(int)

        DATASET_FMA_GENRE_ID = genres_df.set_index('title')['genre_id'].to_dict()
        for genre, id in DATASET_FMA_GENRE_ID.items():
            self.GENRE_TRANSLATOR[genre]['fma'] = id
            self.GENRE_ID_TRANSLATOR['fma'][id] = genre

        raw_tracks_df = pd.read_csv(os.path.join(self.metadata_path,'raw_tracks.csv'))
        tracks = raw_tracks_df[['track_id','track_genres']]

        self.genres_id = tracks.apply(lambda x: self.extract_genre_id(x), axis=1)
        self.genres_id = pd.DataFrame(self.genres_id.tolist(), columns=['track_id', 'genre_id','folder','file'])
        self.genres_id['track_id'] = self.genres_id['track_id'].fillna(-1).astype(int).astype(str).str.zfill(6)
        self.genres_id['folder'] = self.genres_id['track_id'].apply(lambda x: x[0:3])
        self.genres_id = self.genres_id.apply(lambda x: self.check_file_in_folder(x), axis=1, )

        # print(GENRE_TRANSLATOR)
        # print(GENRE_ID_TRANSLATOR)
        # print(GENRE_TRANSLATOR[GENRE_ID_TRANSLATOR['fma'][21]]['system'])
        # print(GENRE_TRANSLATOR[GENRE_ID_TRANSLATOR['fma'][4]]['system'])


        self.genres_id = self.genres_id.dropna(subset=['genre_id'])
        self.genres_id = self.genres_id.reset_index(drop=True)

        self.genres_id = self.filter_dataset_for_genres(self.genres_id,'fma')
        

    def get_genre_map(self, row):
        title = row['title'].lower() 
        if title in self.GENRE_TITLE:
            return (row['genre_id'],title)
        
    def extract_genre_id(sefl, x):
        try:
            return (x['track_id'],json.loads(x['track_genres'].replace("'", '"'))[0]['genre_id'],'','')
        except:
            return None

    def check_file_in_folder(self, row):
        folder = os.path.join(self.files_path,row['folder'])
        track_path = f'''{os.path.join(folder,row['track_id'])}.mp3'''
        row['file'] = track_path
        return row