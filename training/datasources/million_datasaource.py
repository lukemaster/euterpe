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
import json
import pandas as pd
from .datasource import Datasource

from training.config import Config

cfg = Config()

class MillionDatasource(Datasource):

    def __init__(self, datasets_path):
        super().__init__(datasets_path,'million')
        self.dataset_path = os.path.join(self.datasets_path,'million')
        self.metadata_path = self.dataset_path
        self.files_path = cfg.MILLION_PATH


        genres_df = pd.read_csv(os.path.join(self.metadata_path,'Music Info.csv'))
        genres = set(genres_df['genre'])
        genre_map = {}
        for genre in genres:
            if type(genre) == str:
                g_lower = genre.lower()
                if g_lower in self.GENRE_TITLE:
                    genre_map[g_lower] = genre
                    self.GENRE_TRANSLATOR[g_lower]['million'] = g_lower
                    self.GENRE_ID_TRANSLATOR['million'][g_lower] = g_lower
        
        self.genres_id = pd.DataFrame(columns=['file','genre_id'])
        print(self.genres_id)
        rows = []
        for k,folder_name in genre_map.items():
            folder = os.path.join(self.files_path,folder_name)
            for file in os.listdir(folder):
                rows.append({'file':os.path.join(folder,file), 'genre_id':k})
            
        self.genres_id = pd.DataFrame(rows)