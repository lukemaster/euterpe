# Copyright (C) 2025 Rafael Luque Tejada
# Author: Rafael Luque Tejada <lukemaster.master@gmail.com>
#
# This file is part of Generación de Música Personalizada a través de Modelos Generativos Adversariales.
#
# Generación de Música Personalizada a través de Modelos Generativos Adversariales is free software: you can redistribute it and/or modify
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
# import numpy as np
# np.complex = complex  # Corrección temporal necesaria para librosa
import librosa
import librosa.display
########################

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import librosa
import numpy as np
import torch

from training.config import Config

cfg = Config()

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.SAMPLE_RATE = cfg.SAMPLE_RATE
        self.N_FFT = cfg.N_FFT
        self.HOP_LENGTH = cfg.HOP_LENGTH
        self.TOTAL_DURATION = cfg.TOTAL_DURATION
        self.SEGMENT_DURATION = cfg.SEGMENT_DURATION
        self.segment_samples = self.SEGMENT_DURATION * self.SAMPLE_RATE
        adjustment = 0
        if (self.segment_samples + (self.N_FFT - self.HOP_LENGTH)) % self.HOP_LENGTH != 0:
            adjustment = 1
        self.input_w = (self.segment_samples + (self.N_FFT - self.HOP_LENGTH)) // self.HOP_LENGTH + adjustment

        self.file_paths = list(file_paths)
        self.labels = list(labels)
        assert len(self.file_paths) == len(self.labels), "Labels amount don't match with files amount"

        self.segments_per_track = int(self.TOTAL_DURATION / cfg.SEGMENT_DURATION)

        self.num_freq_bins = None

    def spec_transform(self, waveform):
        spec = torchaudio.transforms.Spectrogram(
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            power=2.0
        )(waveform)
        self.num_freq_bins = self.N_FFT // 2 + 1
        return torchaudio.transforms.AmplitudeToDB()(spec)

    def __len__(self):
        return len(self.file_paths) * self.segments_per_track
    
    def jump_next_file(self, index):
        # print('Rolling to next file...')
        del self.file_paths[index]
        return self.__getitem__((index + 1) % len(self.file_paths))
            
    def __getitem__(self, index):
        try:
            segment_samples = self.segment_samples

            file_index = index // self.segments_per_track
            segment_index = index % self.segments_per_track

            filepath = self.file_paths[file_index]
            genre_idx = self.labels[file_index]

            y, sr = librosa.load(filepath, sr=self.SAMPLE_RATE)

            if sr < self.SAMPLE_RATE:
                raise ValueError(f"{filepath} descartado por sample rate bajo: {sr}")

            y = librosa.to_mono(y)
            pad_len = (self.N_FFT - self.HOP_LENGTH) // 2
            y = np.pad(y, pad_width=(pad_len, pad_len), mode='reflect')

            start = segment_index * segment_samples
            end = start + segment_samples
            y_segment = y[start:end]
            y_segment = np.pad(y_segment, pad_width=(pad_len, pad_len), mode='reflect')

            if y_segment.shape[0] < segment_samples:
                y_segment = np.pad(y_segment, pad_width=(0, segment_samples - y_segment.shape[0]))

            self.SPEC_TIME_STEPS = int((segment_samples + 2 * pad_len) / self.HOP_LENGTH) + 1

            D = librosa.stft(y_segment, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
            S = np.abs(D)
            S_db = librosa.amplitude_to_db(S, ref=np.max)

            S_db = np.clip(S_db, cfg.DB_MIN, cfg.DB_MAX)

            segment_spec = torch.tensor(S_db).unsqueeze(0).float()
            segment_spec = (segment_spec - cfg.DB_MIN) / (cfg.DB_MAX - cfg.DB_MIN)
            
            segment_spec = segment_spec * 2 - 1  # [-1, 1]

            if segment_spec.shape[-1] < self.SPEC_TIME_STEPS:
                pad_amount = self.SPEC_TIME_STEPS - segment_spec.shape[-1]
                segment_spec = torch.nn.functional.pad(segment_spec, (0, pad_amount))
            else:
                segment_spec = segment_spec[..., :self.SPEC_TIME_STEPS]

            return segment_spec, genre_idx.clone().detach().to(torch.long)

        except Exception as e:
            print(f"caught exception: {e}")
            return self.jump_next_file(index)
        except RuntimeError as e:
            print(f"[ERROR] Archivo dañado o ilegible: {filepath} - {e}")
            return self.jump_next_file(index)
