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
sys.path.append("/home/luke/VIU/09MIAR/euterpe")

import torch
import librosa
import soundfile as sf
from pydub import AudioSegment
from flask import Flask, send_file

from training.vae.vae import VAE
from training.config import Config

cfg = Config()

OUT_PATH = "static/generated"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "vae_best.pt")


class MusicGenerator:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VAE()
        raw_state = torch.load(model_path, map_location=self.device)
        clean_state = {k.replace("model.", ""): v for k, v in raw_state.items()}
        
        self.model.load_state_dict(clean_state, strict=False)
        self.model.to(self.device)
        self.model.eval()


    def eval(self) -> None:
        self.model.eval()

    def generate(self, genre: int, dest: str, track_name: str) -> any:
        try:
            genre_tensor = torch.tensor([genre], dtype=torch.long).to(self.device)
            z_shape = (1, self.model.decoder.latent_channels, self.model.decoder.h, self.model.decoder.w)
            noise = torch.randn(z_shape).to(self.device)
            recon = self.model.decoder(noise, genre_tensor).squeeze(0).squeeze(0).cpu()
            recon = recon.detach().cpu().numpy()
            recon_db = (recon + 1) / 2 * (cfg.DB_MAX - cfg.DB_MIN) + cfg.DB_MIN
            spec_db_normalized = (recon_db - cfg.DB_MIN) / (cfg.DB_MAX - cfg.DB_MIN)  # Normalizar a [0, 1]
            recon_amplitude = 10.0 ** (spec_db_normalized / 20.0)  # Convertir dB a amplitud
            
            file_name_wav = os.path.join(BASE_DIR,dest,track_name.replace('mp3','wav'))
            audio = librosa.griffinlim(recon_amplitude, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, win_length=cfg.N_FFT, window='hann', n_iter=100)
            print(file_name_wav)

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
            
            return noise, mp3_io
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)