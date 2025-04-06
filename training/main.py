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
########################

import argparse

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from training.datasources.fma_datasource import FMADatasource
from training.datasets.mp3_validator import MP3ValidatorDataset
from training.datasets.lit_data_module import LitDataModule

from training.gan.gan import GAN
from training.vae.vae import VAE
from training.gan.GAN_AI_model_wrapper import GANAIModelWrapper
from training.vae.VAE_AI_model_wrapper import VAEAIModelWrapper
from training.datasets.audio_dataset import AudioDataset

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("logs/", name="vae_model")

cfg = Config()

device = cfg.device
print(f'Powered by: {device}')

def get_dataloader(datasets_path, valid_files_csv_path):# TODO: integration inside get_data_module
    fma_dataset = FMADatasource(datasets_path)

    file_paths = fma_dataset.get_file_paths()
    labels = fma_dataset.get_labels()
    mp3_validator = MP3ValidatorDataset(file_paths,labels,valid_files_csv_path,cfg.TOTAL_DURATION) #TODO: params
     
    _, dict_dataset = fma_dataset.balanced(mp3_validator.getValidFiles() ,cfg.LIMIT_FILES)

    file_paths = list(dict_dataset.keys())
    labels = [dict_dataset[fp]['label'] for fp in file_paths]

    dataset = AudioDataset(file_paths, labels)
    print('dataset done')
    
    dataloader = DataLoader(dataset, cfg.TRAIN_BATCH_SIZE, shuffle=False, drop_last=True, num_workers=1)
    print('dataloader done')

    return dataloader, dataset

def get_data_module(datasets_path, valid_files_csv_path):
    _, dataset = get_dataloader(datasets_path, valid_files_csv_path)

    data_module = LitDataModule(
        train_dataset=dataset,
        val_split=0.2,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        num_workers=32
    )

    return data_module

def train_gan(datasets_path, valid_files_csv_path):
    model_base = GAN()

    model = GANAIModelWrapper(model_base)

    data_module = get_data_module(datasets_path,valid_files_csv_path)

    trainer = Trainer(
        max_epochs=cfg.TRAIN_EPOCHS,
        accelerator='auto',
        log_every_n_steps=1,
        enable_progress_bar=True
    )
    trainer.fit(model, datamodule=data_module)

def train_vae(datasets_path, valid_files_csv_path):

    model_base = VAE()

    model = VAEAIModelWrapper(model_base)

    data_module = get_data_module(datasets_path,valid_files_csv_path)

    trainer = Trainer(
        max_epochs=cfg.TRAIN_EPOCHS,
        accelerator='auto',
        log_every_n_steps=1,
        enable_progress_bar=True
    )
    trainer.fit(model, datamodule=data_module)

def generate_audio_vae(model_path, genre_id):
    model_base = VAE()
    model = VAEAIModelWrapper(model_base)

    checkpoint = torch.load(model_path, map_location="cuda")
    model.load_state_dict(checkpoint["model"], strict=False)

    model.to("cuda")
    model.eval()

    
    model.generate_audio_from_noise(genre_id, output_path=f'''sample_genre{genre_id}.wav''')
    datasets_path = '/home/luke/VIU/09MIAR/datasets'
    valid_files_csv_path = '/home/luke/valid_files.csv'

    dataloader, dataset = get_dataloader(datasets_path,valid_files_csv_path)
    
    model.generate_audio_from_mu(next(iter(dataloader))[0], genre_id, output_path=f'''sample_genre{genre_id}_mu.wav''')
    model.generate_audio_from_scaled_z(next(iter(dataloader))[0], genre_id, output_path=f'''sample_genre{genre_id}_mu.wav''')
    sample1 = next(iter(dataloader))[0]  # género 0
    sample2 = next(iter(dataloader))[0]  # género 1

    model.interpolate_between_samples(sample1, 0, sample2, 1, alpha=0.1)
    sample, _ = next(iter(dataloader))
    model.generate_audio_from_scaled_z(sample, genre_id=0, scale=4.0)
    model.generate_audio_from_scaled_z(sample, genre_id=0, scale=3.0)
    model.generate_audio_from_scaled_z(sample, genre_id=0, scale=2.0)
    model.generate_audio_from_scaled_z(sample, genre_id=0, scale=1.0)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example usage: python main.py datasets_path valid_files_csv_path model_to_train')
    parser.add_argument('datasets_path', type=str, help='location of the dataset')
    parser.add_argument('valid_files_csv_path', type=str, help='location of the CSV file with valid dataset entries')
    parser.add_argument('model_to_train', type=str, help='model to train: vae or gan')

    args = parser.parse_args()
    model_to_train = args.model_to_train

    if model_to_train != 'vae':
        train_gan(args.datasets_path, args.valid_files_csv_path)
    else:
        train_vae(args.datasets_path, args.valid_files_csv_path)
