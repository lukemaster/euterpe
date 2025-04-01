import os
import ast
import psutil
import csv as csv

import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ValidatorDataset(Dataset):
    def __init__(self,valid_files_csv_path, desired_duration):
        super().__init__()
        self.valid_files_csv_path = valid_files_csv_path
        self.desised_duration = desired_duration
    
    def getValidFiles(self):
        i = 0
        dict_dataset = {}
        
        if not os.path.exists(self.valid_files_csv_path): # Preprocessing valid files csv

            dataloader = DataLoader(self, 1, shuffle=False, num_workers=0, pin_memory=True, pin_memory_device='cuda')
        
            fieldnames = ['filepath', 'label', 'duration']
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
                                    'duration': duration.item()
                                })
                            
                            dict_dataset[file_path[0]] = {
                                'label': label,
                                'duration': duration,
                            }
                            # print(f"file {file_path[0]} added to valid_files", end='\r')
                except StopIteration as e:
                    pass

        else: # valid files csv found
            df_loaded = pd.read_csv(self.valid_files_csv_path,sep=';')
            
            for _, row in df_loaded.iterrows():
                dict_dataset[row['filepath']] = {
                    'label': torch.tensor(row['label'], dtype=torch.long),
                    'duration': torch.tensor(row['duration'], dtype=torch.long)
                }

        return dict_dataset