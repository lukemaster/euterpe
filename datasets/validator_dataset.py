import os
import ast

import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import psutil
class ValidatorDataset(Dataset):
    def __init__(self,valid_files_csv_path, hop_length, sample_rate, desired_duration):
        super().__init__()
        self.valid_files_csv_path = valid_files_csv_path
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.desised_duration = desired_duration
        
        self.process = psutil.Process(os.getpid())

    def print_memory(self,prefix=""):
        mem = self.process.memory_info().rss / (1024 * 1024)
        print(f"{prefix} RAM used: {mem:.2f} MB")
    
    def getValidFiles(self):
        i = 0
        valid_files = []
        dict_dataset = {}
        # Check if valid files csv has been preprocessed before
        if not os.path.exists(self.valid_files_csv_path):
            # Preprocessing valid files csv

            dataloader = DataLoader(self, 1, shuffle=False, num_workers=0, pin_memory=True, pin_memory_device='cuda')

            import csv as csv
            import gc

            fieldnames = ['filepath', 'label', 'duration', 'hop_length', 'sample_rate']
            with open(self.valid_files_csv_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
            with torch.no_grad():
                try:
                    for spectrogram, label, file_path, duration in dataloader:
                        if spectrogram and file_path is not None and duration is not None: 
                            with open(self.valid_files_csv_path, mode='a', newline='') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

                                writer.writerow({
                                    'filepath': file_path[0],
                                    'label': label.item(),
                                    'duration': duration.item(),
                                    'hop_length': self.hop_length,
                                    'sample_'
                                    'rate': self.sample_rate
                                })
                            
                            # print(f"file {file_path[0]} added to valid_files", end='\r')


                            dict_dataset[file_path[0]] = {
                                'label': label,
                                'duration': duration,
                                'hop_length': self.hop_length,
                                'sample_rate': self.sample_rate
                            }
                except StopIteration as e:
                    pass

        else: # valid files csv found
            df_loaded = pd.read_csv(self.valid_files_csv_path,sep=';')
            
            # Convert string representations back to lists
            # df_loaded['label'] = df_loaded['label'].apply(ast.literal_eval)
            # df_loaded['duration'] = df_loaded['duration'].apply(ast.literal_eval)

            # Convert back to dictionary format
            
            for _, row in df_loaded.iterrows():

                dict_dataset[row['filepath']] = {
                    'label': torch.tensor(row['label'], dtype=torch.long),
                    'duration': torch.tensor(row['duration'], dtype=torch.long),
                    'hop_length': row['hop_length'],
                    'sample_rate': row['sample_rate'],
                }

        return dict_dataset