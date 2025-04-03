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


## KEEP IT IN A BLOCK ##
import numpy as np

from training.config import Config
np.complex = complex  # Corrección temporal necesaria para librosa
import librosa
import librosa.display
########################

import os
import torch
from mutagen.mp3 import MP3

from .validator_dataset import ValidatorDataset

cfg = Config()

class MP3ValidatorDataset(ValidatorDataset):
    def __init__(self, file_paths, labels, valid_files_csv_path, cut_duration):
        super().__init__(valid_files_csv_path, cut_duration)
        self.SAMPLE_RATE = cfg.SAMPLE_RATE
        self.N_FFT = cfg.N_FFT
        self.HOP_LENGTH = cfg.HOP_LENGTH
        
        self.file_paths = file_paths
        self.labels = labels
        self.counter = 0
        self.desired_sample_rate = self.SAMPLE_RATE
        self.cut_duration = cut_duration

        # filter invalid files
        print('Creating valid_files')
        self.valid_files = [fp for fp in self.file_paths if self.is_valid_file(fp)]
        print('Done valid_files')

    def __len__(self):
        return len(self.valid_files)
    
    def jump_next_file(self, index):
        # print('Rolling to next file...')
        del self.valid_files[index]
        return self.__getitem__((index + 1) % len(self.file_paths))

    def is_valid_file(self, file_path):
        try:
            audio_info = MP3(file_path).info
            try:
                return audio_info.sample_rate is not None and audio_info.length is not None
            except:
                # print(f'Invalid file: {file_path}: {audio_info.sample_rate} {audio_info.length}')
                return False
        except:
            return False

    def __getitem__(self, idx):
        if idx < self.__len__():
            file_path = self.valid_files[idx]
            self.counter +=1
            
            try:
                audio_info = MP3(file_path).info
                sample_rate = audio_info.sample_rate
                    
                waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True, duration=self.cut_duration)
                if waveform is None or len(waveform) == 0:
                    # print(f'Error: unable to load {file_path}')
                    return self.jump_next_file(idx)
                
                if sample_rate > self.desired_sample_rate:
                    # print(f'Error: resample {file_path} from {sample_rate} to {self.desired_sample_rate}.')
                    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.desired_sample_rate )
                    sample_rate = self.desired_sample_rate 
                elif sample_rate < self.SAMPLE_RATE:
                        return self.jump_next_file(idx)

                num_samples = int(sr * self.cut_duration)
                waveform = waveform[:num_samples]
                
                mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=cfg.NUM_MELS, hop_length=self.HOP_LENGTH)
                mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

                if len(mel_spectrogram.shape) == 0:
                    # print('Error: spectrograma obtained has only one scalar value.')
                    return self.jump_next_file(idx)
                elif len(mel_spectrogram.shape) == 1:
                    # print(f'Error: spectrograma obtained is an 1D vector with shape {mel_spectrogram.shape}.')
                    return self.jump_next_file(idx)
                # elif len(mel_spectrogram.shape) == 2:
                #     pass # all right
                elif len(mel_spectrogram.shape) == 3:
                    # print(f'Warning: spectrograma obtainerd has an extra dimension {mel_spectrogram.shape}')
                    mel_spectrogram = np.squeeze(mel_spectrogram)
                    # print(f'Fixed shape: {mel_spectrogram.shape}')

                if mel_spectrogram.shape != (cfg.NUM_MELS, mel_spectrogram.shape[1]):
                    # print(f'Error: unexpected final shape {mel_spectrogram.shape}.')
                    return self.jump_next_file(idx)

                original_duration = audio_info.length
                if original_duration < self.cut_duration:
                    # print(f'Error: file duration less than {self.cut_duration} seconds for file {file_path}: {mel_spectrogram.shape[1]} seconds. Original duration: {original_duration} seconds.')
                    return self.jump_next_file(idx)
                duration = mel_spectrogram.shape[1]

                label = self.labels[idx]
                # Ensure labels are integers
                label = int(label) if isinstance(label, str) else label
                
                # return mel_spectrogram, torch.tensor(label, dtype=torch.long), file_path, duration#, hop_length, sample_rate
                return True, label, file_path, duration#, hop_length, sample_rate

            except Exception as e:
                # print(f'Error processing {file_path}: {e}')
                # import traceback
                # traceback.print_exc()
                return self.jump_next_file(idx)

        raise StopIteration(f'Finished!')