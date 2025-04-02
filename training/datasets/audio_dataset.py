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

        self.kind_of_spectrogram = cfg.KIND_OF_SPECTROGRAM
        self.num_freq_bins = None

        if self.kind_of_spectrogram != "MEL":
            def stft(waveform):
                spec = torchaudio.transforms.Spectrogram(
                    n_fft=self.N_FFT,
                    hop_length=self.HOP_LENGTH,
                    power=2.0
                )(waveform)
                self.num_freq_bins = self.N_FFT // 2 + 1
                return torchaudio.transforms.AmplitudeToDB()(spec)
            self.spec_transform = stft
            
        else:
            def mel(waveform):
                spec = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.SAMPLE_RATE,
                    n_fft=self.N_FFT,
                    hop_length=self.HOP_LENGTH,
                    n_mels=cfg.NUM_MELS
                )(waveform)
                self.num_freq_bins = cfg.NUM_MELS
                return torchaudio.transforms.AmplitudeToDB()(spec)
            self.spec_transform = mel
            


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

            if self.kind_of_spectrogram == "STFT":
                D = librosa.stft(y_segment, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
                S = np.abs(D)
                S_db = librosa.amplitude_to_db(S, ref=np.max)
            else:
                S = librosa.feature.melspectrogram(
                    y=y_segment,
                    sr=self.SAMPLE_RATE,
                    n_fft=self.N_FFT,
                    hop_length=self.HOP_LENGTH,
                    n_mels=cfg.NUM_MELS
                )
                S_db = librosa.amplitude_to_db(S, ref=np.max)

            segment_spec = torch.tensor(S_db).unsqueeze(0).float()
            spec_mean = segment_spec.mean()
            spec_std = segment_spec.std()

            if spec_std < 1e-5:
                segment_spec = torch.zeros_like(segment_spec)
            else:
                segment_spec = (segment_spec - spec_mean) / spec_std

            if segment_spec.shape[-1] < self.SPEC_TIME_STEPS:
                pad_amount = self.SPEC_TIME_STEPS - segment_spec.shape[-1]
                segment_spec = torch.nn.functional.pad(segment_spec, (0, pad_amount))
            else:
                segment_spec = segment_spec[..., :self.SPEC_TIME_STEPS]

            # print(f'''audio_dataset spectrogram output shape {segment_spec.shape}''')
            # Si el espectrograma es STFT, se usa el número real de bins; no se fuerza a NUM_MELS
            return segment_spec, genre_idx.clone().detach().to(torch.long)

        except Exception as e:
            print(f"caught exception: {e}")
            return self.jump_next_file(index)
        except RuntimeError as e:
            print(f"[ERROR] Archivo dañado o ilegible: {filepath} - {e}")
            return self.jump_next_file(index)