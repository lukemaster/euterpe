class Datasource():
    def __init__(self, datasets_path, datasource_name):
        self.datasets_path = datasets_path
        self.datasource_name = datasource_name
        self.GENRE_TITLE = ['hip-hop','jazz','rock','pop','blues']
        self.GENRE_TRANSLATOR = {
            'hip-hop': {
                'system': 0,
                'fma': None,
                'million': None,
                'jamendo': None,
            }, 
            'jazz': {
                'system': 1,
                'fma': None,
                'million': None,
                'jamendo': None,
            },
            'rock': {
                'system': 2,
                'fma': None,
                'million': None,
                'jamendo': None,
            },
            'pop': {
                'system': 3,
                'fma': None,
                'million': None,
                'jamendo': None,
            },
            'blues': {
                'system': 4,
                'fma': None,
                'million': None,
                'jamendo': None,
            }
        }
        self.GENRE_ID_TRANSLATOR = {
            'fma': {},
            'million': {},
            'jamendo': {},
        }
        self.GENRE_IDS = {
            'hip-hop': 0, 
            'jazz': 1, 
            'rock': 2,
            'pop': 3,
            'blues': 4
        }
        self.genres_id = None
        self.file_paths = None
        self.labels = None

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
        minAmount = 0 # minimal common amount for all genres
        for gen,label in self.GENRE_IDS.items():
            tracks_by_label[label] = self.getTracksByLabel(dict_dataset,label)
            tracks_by_genre[gen] = len(tracks_by_label[label])
            minAmount = max(minAmount,tracks_by_genre[gen])

        minAmount = limit if limit else minAmount

        # getting balance dataset (based on minAmount)
        balanced_dict_dataset = {}
        for _,label in self.GENRE_IDS.items():
            keys = list(tracks_by_label[label].keys())[:minAmount]
            for k in keys:
                balanced_dict_dataset[k] = dict_dataset[k]

        return minAmount, balanced_dict_dataset