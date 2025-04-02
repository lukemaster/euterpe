import os
from dotenv import load_dotenv

import torch

load_dotenv()
load_dotenv('./VIU/09MIAR/euterpe/.env') #TODO: REMOVE PATH

class Config(object):

    _instance = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    KIND_OF_SPECTROGRAM = os.environ.get('KIND_OF_SPECTROGRAM','MEL').upper()
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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.__init_config__()
        return cls._instance

    def __init_config__(self):
        if self.KIND_OF_SPECTROGRAM != 'MEL': #STFT
            self.SEGMENT_DURATION=2
            self.N_FFT = 1024
            self.HOP_LENGTH = self.N_FFT // 4
            self.SPEC_ROWS = self.N_FFT // 2 + 1
            self.SPEC_COLS = int(self.SAMPLE_RATE * self.SEGMENT_DURATION / self.HOP_LENGTH)
        else: #MEL
            self.SEGMENT_DURATION=5
            self.N_FFT = 1024
            self.HOP_LENGTH = int(os.environ.get('HOP_LENGTH'))
            self.SPEC_ROWS = 128
            self.SPEC_COLS = int(self.SAMPLE_RATE * self.SEGMENT_DURATION / self.HOP_LENGTH)
            self.NUM_MELS = 128
        self.SPEC_TIME_STEPS = int(self.SAMPLE_RATE * self.SEGMENT_DURATION / self.HOP_LENGTH)

        self.GENRE_EMBED_DIM = int(os.environ.get('GENRE_EMBED_DIM'))
        self.GEN_MODEL_DIM = int(os.environ.get('GEN_MODEL_DIM'))
        self.GEN_NUM_LAYERS = int(os.environ.get('GEN_NUM_LAYERS'))
        self.GEN_NUM_HEADS = int(os.environ.get('GEN_NUM_HEADS'))
        self.GAN_PRETRAIN_EPOCHS_D = int(os.environ.get('GAN_PRETRAIN_EPOCHS_D'))
        self.GAN_PRETRAIN_EPOCHS_G = int(os.environ.get('GAN_PRETRAIN_EPOCHS_G'))

        self.BETA_MAX = float(os.environ.get('BETA_MAX'))
        self.BETA_WARMUP_EPOCHS = int(os.environ.get('BETA_WARMUP_EPOCHS'))