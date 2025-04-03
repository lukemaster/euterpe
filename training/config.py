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
from dotenv import load_dotenv

import torch

load_dotenv()
load_dotenv('./VIU/09MIAR/euterpe/.env') #TODO: REMOVE PATH

class Config(object):

    _instance = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SAMPLE_RATE = 44100
    NUM_GENRES = int(os.environ.get('NUM_GENRES'))
    LATENT_DIM = int(os.environ.get('LATENT_DIM'))
    TOTAL_DURATION = 25
    TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
    LIMIT_FILES = int(os.environ.get('LIMIT_FILES'))
    TRAIN_EPOCHS = int(os.environ.get('TRAIN_EPOCHS'))
    
    # DATASETS
    FMA_PATH = os.environ.get('FMA_PATH')
    MILLION_PATH = os.environ.get('MILLION_PATH')
    JAMENDO_PATH = os.environ.get('JAMENDO_PATH')


    # other config
    # CNN=[1,32,64,128] # be careful: it must always starts with 1
    # CNN_KERNEL=4
    # CNN_STRIDE=2
    # CNN_PADDING=1
    # LATENT_CHANNELS = 16
    # LSTM_HIDDEN_SIZE = 256#128
    # LSTM_NUM_LAYERS = 2

    CNN = [1, 64, 128, 256] # be careful: it must always starts with 1
    LATENT_CHANNELS = 32
    LSTM_HIDDEN_SIZE = 512
    CNN_KERNEL = 3
    CNN_STRIDE = 2
    CNN_PADDING = 1
    LSTM_NUM_LAYERS = 2

    debug = False

    GENRE_EMBED_DIM=8

    DB_MIN = int(os.environ.get('DB_MIN'))
    DB_MAX = int(os.environ.get('DB_MAX'))


    LEARNING_RATE = 0.0001
    LR_SCHEDULER_PATIENTE = 6
    LR_SCHEDULER_FACTOR = 0.5

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.__init_config__()
        return cls._instance

    def __init_config__(self):
        self.SEGMENT_DURATION=10
        self.N_FFT = 2048
        self.HOP_LENGTH = self.N_FFT // 4
        self.SPEC_ROWS = self.N_FFT // 2 + 1
        self.SPEC_COLS = int(self.SAMPLE_RATE * self.SEGMENT_DURATION / self.HOP_LENGTH)
        self.SPEC_TIME_STEPS = int(self.SAMPLE_RATE * self.SEGMENT_DURATION / self.HOP_LENGTH)

        self.GEN_MODEL_DIM = int(os.environ.get('GEN_MODEL_DIM'))
        self.GEN_NUM_LAYERS = int(os.environ.get('GEN_NUM_LAYERS'))
        self.GEN_NUM_HEADS = int(os.environ.get('GEN_NUM_HEADS'))
        self.GAN_PRETRAIN_EPOCHS_D = int(os.environ.get('GAN_PRETRAIN_EPOCHS_D'))
        self.GAN_PRETRAIN_EPOCHS_G = int(os.environ.get('GAN_PRETRAIN_EPOCHS_G'))
        self.GAN_BETA_MIN = int(os.environ.get('GAN_BETA_MIN'))
        self.GAN_BETA_MAX = int(os.environ.get('GAN_BETA_MAX'))

        self.BETA_MAX = float(os.environ.get('BETA_MAX'))
        self.BETA_WARMUP_EPOCHS = int(os.environ.get('BETA_WARMUP_EPOCHS'))