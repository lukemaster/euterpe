## KEEP IT IN A BLOCK ##
import numpy as np
np.complex = complex  # Corrección temporal necesaria para librosa
########################

import os
import argparse

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from gan.gan import LitGAN
from vae.vae import LitVAE
from datasets.audio_dataset import AudioDataset
from datasources.fma_datasource import FMADatasource
from datasets.mp3_validator import MP3ValidatorDataset
from datasets.lit_data_module import VAEDataModule

from dotenv import load_dotenv
load_dotenv('./VIU/09MIAR/euterpe/.env')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Powered by: {device}')

TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))

def get_dataloader(datasets_path, valid_files_csv_path, num_mels):# TODO: integration inside get_data_module
    fma_dataset = FMADatasource(datasets_path)

    file_paths = fma_dataset.get_file_paths()
    labels = fma_dataset.get_labels()
    mp3Validator = MP3ValidatorDataset(file_paths,labels,valid_files_csv_path,num_mels,10,25,int(os.environ.get('SAMPLE_RATE'))) #TODO: params
     
    _, dict_dataset = fma_dataset.balanced(mp3Validator.getValidFiles() ,int(os.environ.get('LIMIT_FILES')))

    file_paths = list(dict_dataset.keys())
    labels = [dict_dataset[fp]['label'] for fp in file_paths]

    dataset = AudioDataset(file_paths, labels)
    print('dataset done')
    
    dataloader = DataLoader(dataset, TRAIN_BATCH_SIZE, shuffle=False, drop_last=True, num_workers=30)
    print('dataloader done')

    return dataloader, dataset

def get_data_module(datasets_path, valid_files_csv_path):
    _, dataset = get_dataloader(datasets_path, valid_files_csv_path, int(os.environ.get('NUM_MELS')))

    data_module = VAEDataModule(
        train_dataset=dataset,
        val_split=0.2,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=32
    )

    return data_module

def train_gan(datasets_path, valid_files_csv_path):
    model = LitGAN()

    data_module = get_data_module(datasets_path,valid_files_csv_path)

    trainer = Trainer(
        max_epochs=int(os.environ.get('TRAIN_EPOCHS')),
        accelerator='auto',
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True
    )
    trainer.fit(model, datamodule=data_module)


def train_vae(datasets_path, valid_files_csv_path):

    model = LitVAE()

    data_module = get_data_module(datasets_path,valid_files_csv_path)

    trainer = Trainer(
        max_epochs=int(os.environ.get('TRAIN_EPOCHS')),
        accelerator='auto',
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True
    )
    trainer.fit(model, datamodule=data_module)


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
