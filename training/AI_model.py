import os
import json
import numpy as np

import torch
import torch.nn.functional as F

import librosa

import pytorch_lightning as pl

import matplotlib.pyplot as plt

from training.config import Config

cfg = Config()

class AIModel(pl.LightningModule):
    BETA_MAX = cfg.BETA_MAX
    BETA_WARMUP_EPOCHS = cfg.BETA_WARMUP_EPOCHS
    LATENT_DIM = cfg.LATENT_DIM
    SAMPLE_RATE = cfg.SAMPLE_RATE
    N_FFT = cfg.N_FFT
    HOP_LENGTH = cfg.HOP_LENGTH
    SPEC_TIME_STEPS = cfg.SPEC_TIME_STEPS
    
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

    def generate_audio_from_noise(self, genre_id: int, output_path: str = 'generated.wav'):


        device = self.device
        self.eval()
        vae = self.model

        with open(self.genre_db_limits_path, 'r') as f:
            genre_db_limits = json.load(f)
        db_min, db_max = genre_db_limits[str(genre_id)]

        num_segments = int(cfg.TOTAL_DURATION / cfg.SEGMENT_DURATION)
        zs = [torch.randn(1, self.LATENT_DIM).to(device) for _ in range(num_segments)]
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
        print(f'[DEBUG] spec antes de denormalizar: min={spec_min:.4f}, max={spec_max:.4f}')
        if abs(spec_max - spec_min) < 1e-3:
            print('[WARNING] El espectrograma generado tiene un rango demasiado pequeño')

        try:
            mean_x = self.mean_x_buffer.item() if hasattr(self.mean_x_buffer, 'item') else float(self.mean_x_buffer)
            std_x = self.std_x_buffer.item() if hasattr(self.std_x_buffer, 'item') else float(self.std_x_buffer)
        except Exception as e:
            print(f'[ERROR] Acceso a mean_x o std_x: {e}')
            mean_x, std_x = 0.0, 1.0

        if std_x != 0:
            spec = spec * std_x + mean_x

        spec_db = spec * (db_max - db_min) + db_min
        spec_amp = np.power(10.0, spec_db.numpy() / 20.0)

        n_fft = self.N_FFT
        if cfg.KIND_OF_SPECTROGRAM != 'MEL':
            expected_bins = n_fft // 2 + 1
            if spec_amp.shape[0] != expected_bins:
                print(f'[WARNING] Redimensionando de {spec_amp.shape[0]} a {expected_bins} bandas para GriffinLim')
                spec_amp = librosa.util.fix_length(spec_amp, size=expected_bins, axis=0)

            griffin_waveform = librosa.griffinlim(
                spec_amp,
                n_iter=64,
                hop_length=self.HOP_LENGTH,
                win_length=n_fft,
                window='hann'
            )
        else:
            mel_basis = librosa.filters.mel(
                sr=self.SAMPLE_RATE,
                n_fft=n_fft,
                n_mels=cfg.NUM_MELS
            )
            inv_mel_basis = np.linalg.pinv(mel_basis)
            linear_spec = np.dot(inv_mel_basis, spec_amp)

            expected_bins = n_fft // 2 + 1
            if linear_spec.shape[0] != expected_bins:
                print(f'[WARNING] Redimensionando MEL inverso de {linear_spec.shape[0]} a {expected_bins} bandas')
                linear_spec = librosa.util.fix_length(linear_spec, size=expected_bins, axis=0)

            griffin_waveform = librosa.griffinlim(
                linear_spec,
                n_iter=64,
                hop_length=self.HOP_LENGTH,
                win_length=n_fft,
                window='hann'
            )

        plt.figure(figsize=(12, 4))
        plt.imshow(spec_db.numpy(), aspect='auto', origin='lower', cmap='magma')
        plt.title(f'Espectrograma generado (género {genre_id})')
        plt.xlabel('Frames temporales')
        plt.ylabel('Frecuencia')
        plt.colorbar(label='Potencia (dB)')
        plt.tight_layout()
        plt.show()

        import soundfile as sf
        sf.write(output_path, griffin_waveform, self.SAMPLE_RATE)
        print(f'Audio generado guardado en: {output_path}')

        return griffin_waveform, zs
