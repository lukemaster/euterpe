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
import numpy as np

import torch
import torch.nn.functional as F

import librosa
import soundfile as sf

import pytorch_lightning as pl

import matplotlib.pyplot as plt

from training.config import Config

cfg = Config()

class AIModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.register_buffer('x_sum', torch.tensor(0.0))
        self.register_buffer('x_squared_sum', torch.tensor(0.0))
        self.register_buffer('x_count', torch.tensor(0))
        self.std_x = 0
        self.mean_x = 0
        self.best_val_loss = float('inf')

        self.genre_db_limits_path = 'logs/genre_db_limits.json'
        self.genre_db_limits = self.load_or_init_genre_limits()

        self.train_step_metrics = []
        self.val_step_metrics = []
        self.train_epoch_metrics = []
        self.val_epoch_metrics = []

    def on_fit_start(self):
        train_dataloader = self.trainer.datamodule.train_dataloader()
        steps_per_epoch = len(train_dataloader)
        self.total_steps = self.trainer.max_epochs * steps_per_epoch

        if cfg.debug:
            print(f"[DEBUG] Total training steps: {self.total_steps}")

    def load_or_init_genre_limits(self):
        if os.path.exists(self.genre_db_limits_path):
            with open(self.genre_db_limits_path, 'r') as f:
                return json.load(f)
        return {str(i): [100.0, -100.0] for i in range(cfg.NUM_GENRES)}

    def update_genre_limits(self, genre_id, spec):
        spec_db = 20 * torch.log10(torch.clamp(spec, min=1e-5)).cpu().numpy()
        min_db = float(np.min(spec_db))
        max_db = float(np.max(spec_db))
        g = str(genre_id)
        if g not in self.genre_db_limits:
            self.genre_db_limits[g] = [min_db, max_db]
        else:
            current_min, current_max = self.genre_db_limits[g]
            self.genre_db_limits[g][0] = min(current_min, min_db)
            self.genre_db_limits[g][1] = max(current_max, max_db)

    def save_genre_limits(self):
        with open(self.genre_db_limits_path, 'w') as f:
            json.dump(self.genre_db_limits, f, indent=2)

    def generate_spectrogram(self,spec):
        spec = spec.detach().cpu().numpy()
        spec = (spec + 1) / 2 * (cfg.DB_MAX - cfg.DB_MIN) + cfg.DB_MIN

        return spec

    def generate_audio_from_noise(self, genre_id: int, output_path: str = 'generated.wav'):
        device = self.device
        self.eval()
        vae = self.model

        with open(self.genre_db_limits_path, 'r') as f:
            genre_db_limits = json.load(f)
        db_min, db_max = genre_db_limits[str(genre_id)]

        num_segments = int(cfg.TOTAL_DURATION / cfg.SEGMENT_DURATION)
        zs = [torch.randn(1, cfg.LATENT_DIM).to(device) for _ in range(num_segments)]
        genre = torch.tensor([genre_id], dtype=torch.long).to(device)

        with torch.no_grad():
            decoded_specs = [vae.decoder(z, genre)[0, 0].cpu() for z in zs]

        n = 20
        for i in range(len(decoded_specs)):
            if decoded_specs[i].shape[-1] >= n + 1:
                decoded_specs[i][..., :n] = decoded_specs[i][..., n].unsqueeze(-1).expand_as(decoded_specs[i][..., :n])
                decoded_specs[i][..., -n:] = decoded_specs[i][..., -n-1].unsqueeze(-1).expand_as(decoded_specs[i][..., -n:])

        spec = torch.cat(decoded_specs, dim=-1)

        spec_min = spec.min().item()
        spec_max = spec.max().item()
        
        if abs(spec_max - spec_min) < 1e-3:
            print('[WARNING] Generated spectrogram has a very little range')

        try:
            mean_x = self.mean_x_buffer.item() if hasattr(self.mean_x_buffer, 'item') else float(self.mean_x_buffer)
            std_x = self.std_x_buffer.item() if hasattr(self.std_x_buffer, 'item') else float(self.std_x_buffer)
        except Exception as e:
            print(f'[ERROR] access to mean_x o std_x: {e}')
            mean_x, std_x = 0.0, 1.0

        if std_x != 0:
            spec = spec * std_x + mean_x

        spec_db = spec * (db_max - db_min) + db_min
        spec_amp = np.power(10.0, spec_db.numpy() / 20.0)

        n_fft = cfg.N_FFT
        expected_bins = n_fft // 2 + 1
        if spec_amp.shape[0] != expected_bins:
            print(f'[WARNING] Resizing from {spec_amp.shape[0]} to {expected_bins} GriffinLim bands')
            spec_amp = librosa.util.fix_length(spec_amp, size=expected_bins, axis=0)

        griffin_waveform = librosa.griffinlim(
            spec_amp,
            n_iter=32,
            hop_length=cfg.HOP_LENGTH,
            win_length=n_fft,
            window='hann'
        )
        plt.figure(figsize=(12, 4))

        freqs = np.linspace(0, cfg.SAMPLE_RATE // 2, spec_db.shape[0])
        times = np.linspace(0, cfg.TOTAL_DURATION, spec_db.shape[1])

        plt.imshow(spec_db.numpy(), aspect='auto', origin='lower', cmap='magma', extent=[times[0], times[-1], freqs[0], freqs[-1]])

        plt.title(f'Generaged spectrogram - Genre {genre_id}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequence (Hz)')
        cbar = plt.colorbar(label='Power (dB)')
        cbar.ax.tick_params(labelsize=10)

        plt.tight_layout()
        plt.show()

        
        sf.write(output_path, griffin_waveform, cfg.SAMPLE_RATE)
        print(f'Soundfile generated and saved in: {output_path}')

        return griffin_waveform, zs
