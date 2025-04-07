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

    def clean_spec(self, spec, umbral_relativo=0.1):
        print(spec.shape)
        print(type(spec))
        max_col = np.max(spec, axis=0, keepdims=True)
        max_col[max_col == 0] = 1e-8

        mascara = spec >= (max_col * umbral_relativo)

        spec_filtrado = np.where(mascara, spec, 0.0)

        return spec_filtrado
    
    def generate_spectrogram(self,spec):
        spec = spec.detach().cpu().numpy()
        spec = (spec + 1) / 2 * (cfg.DB_MAX - cfg.DB_MIN) + cfg.DB_MIN
        return spec

    def generate_audio_from_noise(self, genre_id: int, output_path: str = 'generated.wav'):
        self.eval()
        with torch.no_grad():
            z_shape = (1, self.model.decoder.latent_channels, self.model.decoder.h, self.model.decoder.w)
            z = torch.randn(z_shape).to(cfg.device)
            genre_tensor = torch.tensor([genre_id], dtype=torch.long).to(cfg.device)

            recon = self.model.decoder(z, genre_tensor).squeeze(0).squeeze(0).cpu()
            spec_recon = self.generate_spectrogram(recon)

            spec_recon = (spec_recon + 1) / 2 * (cfg.DB_MAX - cfg.DB_MIN) + cfg.DB_MIN

            plt.figure(figsize=(10, 4))
            librosa.display.specshow(spec_recon, sr=cfg.SAMPLE_RATE, hop_length=cfg.HOP_LENGTH, x_axis='time', y_axis='linear', cmap='magma')
            plt.colorbar(format="%+2.0f dB")
            plt.title(f'Spectrogram [librosa]')
            plt.tight_layout()
            plt.show()
            plt.figure(figsize=(10, 4))

            
            if cfg.KIND_OF_SPECTROGRAM == 'MEL':
                extent = [0, spec_recon.shape[1], 0, cfg.SAMPLE_RATE // 2]
                plt.imshow(spec_recon, aspect='auto', origin='lower', cmap='magma', extent=extent)
                plt.set_ylabel('Mel scale')
            else:  # STFT
                freqs = np.linspace(0, cfg.SAMPLE_RATE // 2, spec_recon.shape[0])
                extent = [0, spec_recon.shape[1], freqs[0], freqs[-1]]
                plt.imshow(spec_recon, aspect='auto', origin='lower', cmap='magma', extent=extent)
                plt.set_ylabel('Frequency (Hz)')

            plt.set_title('Spectrogram [imshow] (vmin=-80)')
            plt.set_xlabel('Time frames')

            # recon_amplitude = librosa.db_to_amplitude(recon_db)
            recon_amplitude = 10.0 ** (spec_recon / 20.0)
            audio = librosa.griffinlim(recon_amplitude, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, win_length=cfg.N_FFT, window='hann', n_iter=100)
            sf.write(output_path, audio, cfg.SAMPLE_RATE)
            return audio, spec_recon

        
    def generate_audio_from_mu(self, sample: torch.Tensor, genre_id: int, output_path: str = 'recon.wav'):
        self.eval()
        with torch.no_grad():
            x = sample.to(cfg.device)
            genre_tensor = torch.tensor([genre_id], dtype=torch.long).to(cfg.device)
            mu, _ = self.model.encoder(x, genre_tensor)

            expected_w = self.model.decoder.w
            if mu.shape[3] > expected_w:
                mu = mu[:, :, :, :expected_w]

            recon = self.model.decoder(mu, genre_tensor).squeeze(0).squeeze(0).cpu()
            recon_db = self.generate_spectrogram(recon)
            
            # Visualizaciones
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(recon_db, sr=cfg.SAMPLE_RATE, hop_length=cfg.HOP_LENGTH, x_axis='time', y_axis='linear', cmap='magma')
            plt.colorbar(format="%+2.0f dB")
            plt.title(f'Spectrogram [librosa]')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.imshow(recon_db, aspect='auto', origin='lower', cmap='magma', vmin=-80, vmax=0)
            plt.colorbar()
            plt.title('Spectrogram [imshow] (vmin=-80)')
            plt.tight_layout()
            plt.show()

            recon_amplitude = librosa.db_to_amplitude(recon_db)
            audio = librosa.griffinlim(recon_amplitude, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, win_length=cfg.N_FFT, window='hann', n_iter=100)
            sf.write(output_path, audio, cfg.SAMPLE_RATE)
            return audio, recon_db


    def generate_audio_from_scaled_z(self, sample: torch.Tensor, genre_id: int, scale: float = 2.0, output_path: str = 'scaled_z.wav'):
        self.eval()
        with torch.no_grad():
            x = sample.to(cfg.device)
            genre_tensor = torch.tensor([genre_id], dtype=torch.long).to(cfg.device)
            mu, logvar = self.model.encoder(x, genre_tensor)

            expected_w = self.model.decoder.w
            if mu.shape[3] > expected_w:
                mu = mu[:, :, :, :expected_w]
                logvar = logvar[:, :, :, :expected_w]

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + scale * eps * std

            recon = self.model.decoder(z, genre_tensor).squeeze(0).squeeze(0).cpu()
            recon_db = self.generate_spectrogram(recon)

            plt.figure(figsize=(10, 4))
            librosa.display.specshow(recon_db, sr=cfg.SAMPLE_RATE, hop_length=cfg.HOP_LENGTH, x_axis='time', y_axis='linear', cmap='magma')
            plt.colorbar(format="%+2.0f dB")
            plt.title(f'Spectrogram [librosa]')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.imshow(recon_db, aspect='auto', origin='lower', cmap='magma', vmin=-80, vmax=0)
            plt.colorbar()
            plt.title('Spectrogram [imshow] (vmin=-80)')
            plt.tight_layout()
            plt.show()

            recon_amplitude = librosa.db_to_amplitude(recon_db)
            audio = librosa.griffinlim(recon_amplitude, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, win_length=cfg.N_FFT, window='hann', n_iter=100)
            sf.write(output_path, audio, cfg.SAMPLE_RATE)
            return audio, recon_db


    def interpolate_between_samples(self,
                                sample1: torch.Tensor,
                                genre_id1: int,
                                sample2: torch.Tensor,
                                genre_id2: int,
                                alpha: float,
                                output_path: str = 'interp.wav'):
        self.eval()
        with torch.no_grad():
            x1 = sample1.to(cfg.device)
            x2 = sample2.to(cfg.device)
            genre_tensor1 = torch.tensor([genre_id1], dtype=torch.long).to(cfg.device)
            genre_tensor2 = torch.tensor([genre_id2], dtype=torch.long).to(cfg.device)

            mu1, _ = self.model.encoder(x1, genre_tensor1)
            mu2, _ = self.model.encoder(x2, genre_tensor2)

            expected_w = self.model.decoder.w
            if mu1.shape[3] > expected_w:
                mu1 = mu1[:, :, :, :expected_w]
                mu2 = mu2[:, :, :, :expected_w]

            mu_interp = (1 - alpha) * mu1 + alpha * mu2
            genre_tensor_interp = genre_tensor1 if alpha < 0.5 else genre_tensor2

            recon = self.model.decoder(mu_interp, genre_tensor_interp).squeeze(0).squeeze(0).cpu()
            recon_db = self.generate_spectrogram(recon)

            plt.figure(figsize=(10, 4))
            librosa.display.specshow(recon_db,
                                    sr=cfg.SAMPLE_RATE,
                                    hop_length=cfg.HOP_LENGTH,
                                    x_axis='time',
                                    y_axis='linear',
                                    cmap='magma')
            plt.colorbar(format="%+2.0f dB")
            plt.title(f'Interpolated Spectrogram α={alpha:.2f}')
            plt.tight_layout()
            plt.show()

            recon_amplitude = librosa.db_to_amplitude(recon_db)
            audio = librosa.griffinlim(
                recon_amplitude,
                n_fft=cfg.N_FFT,
                hop_length=cfg.HOP_LENGTH,
                win_length=cfg.N_FFT,
                window='hann',
                n_iter=100
            )

            sf.write(output_path, audio, cfg.SAMPLE_RATE)
            return audio, recon_db

    def generate_audio_from_scaled_z(self,
                                 sample: torch.Tensor,
                                 genre_id: int,
                                 scale: float = 2.0,
                                 output_path: str = 'scaled_z.wav'):
        self.eval()
        with torch.no_grad():
            x = sample.to(cfg.device)
            genre_tensor = torch.tensor([genre_id], dtype=torch.long).to(cfg.device)

            mu, logvar = self.model.encoder(x, genre_tensor)

            expected_w = self.model.decoder.w
            if mu.shape[3] > expected_w:
                print(f"[INFO] Trimming z from width {mu.shape[3]} to {expected_w}")
                mu = mu[:, :, :, :expected_w]
                logvar = logvar[:, :, :, :expected_w]
            elif mu.shape[3] < expected_w:
                raise ValueError(f"[ERROR] z width {mu.shape[3]} is smaller than expected {expected_w}")

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + scale * eps * std

            recon = self.model.decoder(z, genre_tensor).squeeze(0).squeeze(0).cpu()
            recon_db = self.generate_spectrogram(recon)

            plt.figure(figsize=(10, 4))
            librosa.display.specshow(recon_db,
                                    sr=cfg.SAMPLE_RATE,
                                    hop_length=cfg.HOP_LENGTH,
                                    x_axis='time',
                                    y_axis='linear',
                                    cmap='magma')
            plt.colorbar(format="%+2.0f dB")
            plt.title(f'Scaled z Spectrogram (λ={scale})')
            plt.tight_layout()
            plt.show()

            recon_amplitude = librosa.db_to_amplitude(recon_db)
            audio = librosa.griffinlim(
                recon_amplitude,
                n_fft=cfg.N_FFT,
                hop_length=cfg.HOP_LENGTH,
                win_length=cfg.N_FFT,
                window='hann',
                n_iter=100
            )

            sf.write(output_path, audio, cfg.SAMPLE_RATE)
            return audio, recon_db