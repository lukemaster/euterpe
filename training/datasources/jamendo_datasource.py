# Copyright (C) 2025 Rafael Luque Tejada
# Author: Rafael Luque Tejada <lukemaster.master@gmail.com>
#
# This file is part of Generación de Música Personalizada a través de Modelos Generativos Adversariales.
#
# Euterpe as a part of the project Generación de Música Personalizada a través de Modelos Generativos Adversariales is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Generación de Música Personalizada a través de Modelos Generativos Adversariales is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import pandas as pd
from .datasource import Datasource

from training.config import Config

cfg = Config()

class JamendoDatasource(Datasource):

    def __init__(self, datasets_path):
        super().__init__(datasets_path,'jamendo')

        def clean_jamendo_csv(row):
            row['TAGS'] = row['TAGS'].split('genre---')[1]
            return row[['TRACK_ID', 'PATH', 'TAGS']]

        raw_30_df = pd.read_csv(os.path.join(cfg.JAMENDO_PATH,'raw_30s.tsv'), sep='\t', encoding='utf-8', on_bad_lines='skip')
        raw_30_df = raw_30_df[raw_30_df['TAGS'].str.startswith('genre---')]
        
        raw_30_df = raw_30_df.apply(lambda x: clean_jamendo_csv(x),axis=1)
        print(raw_30_df)

        self.dataset_path = os.path.join(self.datasets_path,'mtg-jamendo-dataset','data')
        self.metadata_path = os.path.join(self.dataset_path,'jamendo_metadata')
        self.files_path = cfg.JAMENDO_PATH
        print(self.metadata_path)

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

        self.genres_id = self.genres_id.dropna(subset=['genre_id'])
        self.genres_id = self.genres_id.reset_index(drop=True)

        self.genres_id = self.filter_dataset_for_genres(self.genres_id,'fma')
        