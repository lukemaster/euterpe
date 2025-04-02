## KEEP IT IN A BLOCK ##
import numpy as np

from training.config import Config
np.complex = complex  # Corrección temporal necesaria para librosa
########################

import os
import argparse

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from training.datasources.fma_datasource import FMADatasource
from training.datasets.mp3_validator import MP3ValidatorDataset
from training.datasets.lit_data_module import VAEDataModule

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
    mp3Validator = MP3ValidatorDataset(file_paths,labels,valid_files_csv_path,cfg.TOTAL_DURATION) #TODO: params
     
    _, dict_dataset = fma_dataset.balanced(mp3Validator.getValidFiles() ,cfg.LIMIT_FILES)

    file_paths = list(dict_dataset.keys())
    labels = [dict_dataset[fp]['label'] for fp in file_paths]

    dataset = AudioDataset(file_paths, labels)
    print('dataset done')
    
    dataloader = DataLoader(dataset, cfg.TRAIN_BATCH_SIZE, shuffle=False, drop_last=True, num_workers=1)
    print('dataloader done')

    return dataloader, dataset

def get_data_module(datasets_path, valid_files_csv_path):
    _, dataset = get_dataloader(datasets_path, valid_files_csv_path)

    data_module = VAEDataModule(
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
        accumulate_grad_batches=2,
        precision=16,
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
        accumulate_grad_batches=2,
        precision=16,
        enable_progress_bar=True
    )
    trainer.fit(model, datamodule=data_module)

def generate_audio_vae(model_path, genre_id):
    model_base = VAE()
    model = VAEAIModelWrapper(model_base)

    checkpoint = torch.load(model_path, map_location="cuda")
    model.load_state_dict(checkpoint["model"], strict=False)

    # model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.to("cuda")
    model.eval()

    
    model.generate_audio_from_noise(genre_id, output_path=f'''sample_genre{genre_id}.wav''')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ejemplo de llamada: python main.py datasets_path valid_files_csv_path')
    parser.add_argument('datasets_path', type=str, help='localización del dataset')
    parser.add_argument('valid_files_csv_path', type=str, help='localización del csv de ficheros válidos del dataset')
    parser.add_argument('model_to_train', type=str, help='modelo a entrenar: vae o gan')

    args = parser.parse_args()
    model_to_train = args.model_to_train

    if model_to_train != 'vae':
        train_gan(args.datasets_path, args.valid_files_csv_path)
    else:
        train_vae(args.datasets_path, args.valid_files_csv_path)
