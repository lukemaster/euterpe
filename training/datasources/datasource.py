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


class Datasource():
    def __init__(self, datasets_path, datasource_name):
        self.datasets_path = datasets_path
        self.datasource_name = datasource_name
        self.genres_id = None
        self.file_paths = None
        self.labels = None

        self.GENRE_TRANSLATOR = {}
        self.GENRE_ID_TRANSLATOR = {
            'fma': {},
            'million': {},
            'jamendo': {},
        }
        self.GENRE_TITLE = ['hip-hop','jazz','rock','pop','blues']
        self.GENRE_IDS = {}
        for i, genre_title in enumerate(self.GENRE_TITLE):
            self.GENRE_TRANSLATOR[genre_title] = {
                'system': i,
                'fma': None,
                'million': None,
                'jamendo': None,
            }
            self.GENRE_IDS[genre_title] = i

    def filter_dataset_for_genres(self, dataset, dataset_key):
        genre_ids_filtered = list(self.GENRE_ID_TRANSLATOR[dataset_key].keys())
        dataset['genre_id'] = dataset['genre_id'].astype(int)
        dataset = dataset[dataset['genre_id'].isin(genre_ids_filtered)]
        return dataset
    
    def get_genres_id(self):
        return self.genres_id
    
    def get_file_paths(self):
        # if not self.file_paths and self.genres_id: # TODO
        self.file_paths = self.genres_id['file'].to_numpy()
        return self.file_paths

    def get_labels(self):
        # if not self.labels and not self.genres_id: # TODO
        self.labels = list(self.genres_id.apply(lambda x: self.GENRE_TRANSLATOR[self.GENRE_ID_TRANSLATOR[self.datasource_name][x['genre_id']]]['system'],axis=1))
        return self.labels
    
    @staticmethod
    def get_genre_label_by_id(genre_id):
        return Datasource.GENRE_TITLE[genre_id] if genre_id in Datasource.GENRE_TITLE else ''
    
    def getTracksByLabel(self, dict_dataset, label):
        return dict(filter(lambda x: x[1]['label'] == label, dict_dataset.items()))

    def balanced(self, dict_dataset, limit=None):
         # making indexed object by genre and numeric label
        tracks_by_genre = {}
        tracks_by_label = {}
        min_amount = 0 # minimal common amount for all genres
        for gen,label in self.GENRE_IDS.items():
            tracks_by_label[label] = self.getTracksByLabel(dict_dataset,label)
            tracks_by_genre[gen] = len(tracks_by_label[label])
            min_amount = max(min_amount,tracks_by_genre[gen])

        min_amount = limit if limit else min_amount

        # getting balance dataset (based on min_amount)
        balanced_dict_dataset = {}
        for _,label in self.GENRE_IDS.items():
            keys = list(tracks_by_label[label].keys())[:min_amount]
            for k in keys:
                balanced_dict_dataset[k] = dict_dataset[k]

        return min_amount, balanced_dict_dataset