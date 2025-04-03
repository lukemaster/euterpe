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
import csv as csv

import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from training.config import Config

cfg = Config()

class ValidatorDataset(Dataset):
    def __init__(self,valid_files_csv_path, desired_duration):
        super().__init__()
        self.valid_files_csv_path = valid_files_csv_path
        self.desised_duration = desired_duration
    
    def getValidFiles(self):
        dict_dataset = {}
        
        if not os.path.exists(self.valid_files_csv_path): # Preprocessing valid files csv

            dataloader = DataLoader(self, 1, shuffle=False, num_workers=0, pin_memory=True, pin_memory_device=cfg.device)
        
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