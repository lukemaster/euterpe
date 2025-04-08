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


        self.dataset_path = os.path.join(self.datasets_path,'mtg-jamendo-dataset')
        self.metadata_path = os.path.join(self.dataset_path,'data')
        self.files_path = cfg.JAMENDO_PATH

        raw_30_df = pd.read_csv(os.path.join(self.metadata_path,'raw_30s.tsv'), sep='\t', encoding='utf-8', on_bad_lines='skip')
        raw_30_df = raw_30_df[raw_30_df['TAGS'].str.startswith('genre---')]
        
        raw_30_df = raw_30_df.apply(lambda x: self.clean_jamendo_csv(x),axis=1)
        raw_30_df['TAGS'] = raw_30_df['TAGS'].replace('hiphop', 'hip-hop')


        raw_30_df = raw_30_df.where(raw_30_df['TAGS'].isin(self.GENRE_TITLE)).dropna()
        raw_30_df.drop(columns=['TRACK_ID'], inplace=True)
        tags = raw_30_df['TAGS'].unique()
        for tag in tags:
            self.GENRE_TRANSLATOR[tag]['jamendo'] = tag
            self.GENRE_ID_TRANSLATOR['jamendo'][tag] = tag
        
        raw_30_df.rename(columns={'PATH': 'file'}, inplace=True)
        raw_30_df.rename(columns={'TAGS': 'genre_id'}, inplace=True)

        self.genres_id = raw_30_df

    def clean_jamendo_csv(self, row):
            row['TAGS'] = row['TAGS'].lower().split('genre---')[1]
            row['PATH'] = os.path.join(cfg.JAMENDO_PATH,row['PATH'])
            return row[['TRACK_ID', 'PATH', 'TAGS']]