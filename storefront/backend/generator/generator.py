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

import io
import os
import sys

import numpy as np

# Añadir el directorio raíz del proyecto al PYTHONPATH
import sys

from training.gan.GAN_AI_model_wrapper import GANAIModelWrapper
from training.gan.gan import GAN
sys.path.append("/home/luke/VIU/09MIAR/euterpe")

import torch
import librosa
import soundfile as sf
from pydub import AudioSegment
from flask import Flask, send_file
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import lfilter

from training.vae.vae import VAE
from training.config import Config

cfg = Config()

OUT_PATH = "static/generated"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "gan_best.pt")

class MusicGenerator:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GANAIModelWrapper(GAN(), is_eval=True).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        filtered = {k: v for k, v in checkpoint.items() if k in self.model.state_dict() and v.shape == self.model.state_dict()[k].shape}
        self.model.load_state_dict(filtered, strict=False)
        self.model.eval()


    def eval(self) -> None:
        self.model.eval()

    def normalizar_audio(self, audio, target_dbfs=-20.0):
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (target_dbfs / 20) / (rms + 1e-9)
        return audio * scalar

    def generate(self, genre: int, dest: str, track_name: str) -> any:
        try:
            single_genre_tensor = torch.tensor([genre], device=self.device)
            z_list = []
            single_genre_tensor = torch.tensor([genre], device=self.device)
            zi = GANAIModelWrapper.generate_z_for_genre(single_genre_tensor, cfg)
            z_list.append(zi)

            z = torch.cat(z_list, dim=0)
            genre_tensor = torch.full((1,), genre, device=self.device, dtype=torch.long)

            with torch.no_grad():
                specs = self.model(z, genre_tensor).cpu()

            spec = specs[0][0].numpy()
            spec_db = ((spec + 1.0) / 2.0) * (cfg.DB_MAX - cfg.DB_MIN) * 0.95 + cfg.DB_MIN
            spec_db = np.clip(spec_db, cfg.DB_MIN, cfg.DB_MAX)

            # === Filtro gaussiano + mediana horizontal
            spec_db = gaussian_filter(spec_db, sigma=(1, 1))
            spec_db = median_filter(spec_db, size=(1, 3))

            # === Supresión por energía
            frame_energy = (spec_db ** 2).mean(axis=0)
            mask = frame_energy > np.percentile(frame_energy, 5)
            spec_db = spec_db[:, mask]


            # === Reconstrucción de audio
            magnitude = librosa.db_to_amplitude(spec_db)

            if cfg.KIND_OF_SPECTROGRAM == 'MEL':
                audio = librosa.feature.inverse.mel_to_audio(
                    magnitude,
                    sr=cfg.SAMPLE_RATE,
                    n_fft=cfg.N_FFT,
                    hop_length=cfg.HOP_LENGTH,
                    win_length=cfg.N_FFT,
                    window="hann",
                    center=True,
                    n_iter=512
                )
            else:
                audio = librosa.griffinlim(
                    magnitude,
                    n_iter=512,
                    hop_length=cfg.HOP_LENGTH,
                    win_length=cfg.N_FFT,
                    window="hann",
                    center=True
                )

            # === Normalización, suavizado, post-énfasis y estéreo
            audio = audio / np.max(np.abs(audio))
            audio = self.normalizar_audio(audio, -20)
            audio = lfilter(np.ones(5) / 5.0, 1, audio)
            audio = lfilter([1, -0.97], [1], audio)
            audio_stereo = np.stack([audio, audio], axis=0).T
            audio = audio_stereo

            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()

            audio = np.squeeze(audio)
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
            audio = audio.astype(np.float32)

            wav_io = io.BytesIO()
            sf.write(wav_io, audio, cfg.SAMPLE_RATE, format="WAV")
            wav_io.seek(0)

            audio_segment = AudioSegment.from_file(wav_io, format="wav")
            mp3_io = io.BytesIO()
            audio_segment.export(mp3_io, format="mp3")
            mp3_io.seek(0)
            
            return z, mp3_io
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)