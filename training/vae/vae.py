import os
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

import pytorch_lightning as pl

import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv('./VIU/09MIAR/euterpe/.env')

class VAE(nn.Module):
    NUM_GENRES = int(os.environ.get('NUM_GENRES'))
    GENRE_EMBEDDING_DIM = (2 ** math.ceil(math.log2(NUM_GENRES)))# * 2 maybe an improvement or not

    class VAE_Encoder(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.NUM_GENRES = vae.NUM_GENRES
            self.LATENT_DIM = vae.LATENT_DIM
            self.GENRE_EMBEDDING_DIM = vae.GENRE_EMBEDDING_DIM
            self.SPEC_TIME_STEPS = vae.SPEC_TIME_STEPS
            self.genre_embedding = nn.Embedding(self.NUM_GENRES, self.GENRE_EMBEDDING_DIM)

            self.conv1 = nn.Conv2d(1 + self.GENRE_EMBEDDING_DIM, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)

            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)

            self.pool = nn.MaxPool2d(kernel_size=2)

            
            NUM_MELS = int(os.environ.get('NUM_MELS'))
            with torch.no_grad():
                dummy_spec = torch.zeros(1, 1, NUM_MELS, self.SPEC_TIME_STEPS)
                dummy_genre = torch.zeros(1, dtype=torch.long)
                dummy_emb = self.genre_embedding(dummy_genre).view(1, self.GENRE_EMBEDDING_DIM, 1, 1).expand(-1, -1, NUM_MELS, self.SPEC_TIME_STEPS)
                dummy_input = torch.cat([dummy_spec, dummy_emb], dim=1)

                dummy = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
                dummy = self.pool(F.relu(self.bn2(self.conv2(dummy))))
                dummy = self.pool(F.relu(self.bn3(self.conv3(dummy))))

                self.flattened_shape = dummy.shape[1:]
                self.feature_dim = dummy.numel()

            self.fc_mu = nn.Linear(self.feature_dim, self.LATENT_DIM)
            self.fc_logvar = nn.Linear(self.feature_dim, self.LATENT_DIM)

        def forward(self, x, genre):
            x = x[..., :self.SPEC_TIME_STEPS]
            genre_emb = self.genre_embedding(genre).view(genre.size(0), self.GENRE_EMBEDDING_DIM, 1, 1)
            genre_emb = genre_emb.expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, genre_emb], dim=1)

            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)

            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            
            return mu, logvar

    class VAE_Decoder(nn.Module):
        def __init__(self, vae, feature_dim, flattened_shape):
            super().__init__()
            self.NUM_GENRES = vae.NUM_GENRES
            self.GENRE_EMBEDDING_DIM = vae.GENRE_EMBEDDING_DIM
            self.LATENT_DIM = vae.LATENT_DIM
            self.feature_dim = feature_dim
            self.flattened_shape = flattened_shape

            self.genre_embedding = nn.Embedding(self.NUM_GENRES, self.GENRE_EMBEDDING_DIM)
            self.fc = nn.Linear(self.LATENT_DIM + self.GENRE_EMBEDDING_DIM, feature_dim)

            self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

            self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

            self.output_pad = nn.ConstantPad2d((0, 5), 0)

        def forward(self, z, genre):
            genre_emb = self.genre_embedding(genre)
            z = torch.cat([z, genre_emb], dim=1)

            x = self.fc(z)
            x = x.view(z.size(0), *self.flattened_shape)


            x = x.permute(0, 2, 1, 3)
            B, H, C, W = x.shape
            x = x.reshape(B * H, C, W).permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = x.permute(0, 2, 1).reshape(B, H, C, W)
            x = x.permute(0, 2, 1, 3)

            x = F.relu(self.deconv1(x))
            x = F.relu(self.deconv2(x))
            
            x = self.deconv3(x)
            x = self.output_pad(x)

            
            # if not self.training:
            #     import matplotlib.pyplot as plt

            #     flat = x.detach().cpu().flatten()
            #     x_min = flat.min().item()
            #     x_max = flat.max().item()
            #     x_mean = flat.mean().item()
            #     x_std = flat.std().item()
            #     sat_low = (flat <= -0.95).float().mean().item()
            #     sat_high = (flat >= 0.95).float().mean().item()

            #     print(f"[DEBUG] Decoder output stats -> min: {x_min:.4f}, max: {x_max:.4f}, mean: {x_mean:.4f}, std: {x_std:.4f}")
            #     print(f"[DEBUG] Saturación -> <= -0.95: {sat_low:.2%}, >= 0.95: {sat_high:.2%}")

            #     plt.figure(figsize=(6, 4))
            #     plt.hist(flat.numpy(), bins=50, color='purple', alpha=0.75)
            #     plt.title("Histograma de valores del decoder")
            #     plt.xlabel("Valor")
            #     plt.ylabel("Frecuencia")
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.show()

            return x
            
    def __init__(self):
        super().__init__()
        self.LATENT_DIM = int(os.environ.get('LATENT_DIM'))
        self.SPEC_TIME_STEPS = int((int(os.environ["SAMPLE_RATE"]) * int(os.environ.get('SEGMENT_DURATION'))) / int(os.environ["HOP_LENGTH"]))
        self.encoder = self.VAE_Encoder(self)
        self.decoder = self.VAE_Decoder(self,self.encoder.feature_dim, self.encoder.flattened_shape)
        self.apply(self.init_weights)
        
    def init_weights(_,m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.1, 0.1)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, genre):
        # print(f"[ENCODER IN] x mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        mu, logvar = self.encoder(x, genre)
        # print(f"[LATENT] mu mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")
        # print(f"[LATENT] logvar mean: {logvar.mean().item():.4f}, std: {logvar.std().item():.4f}")

        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, genre)
        if not torch.isfinite(x_hat).all():
            print("[ERROR] x_hat contiene NaNs")

        # print(f"[DECODER OUT] x_recon mean: {x_hat.mean().item():.4f}, std: {x_hat.std().item():.4f}")


        if not torch.isfinite(x).all():
            print("[ERROR] Input contiene NaNs o infinitos")
        if not torch.isfinite(mu).all():
            print("[ERROR] mu contiene NaNs")
        if not torch.isfinite(logvar).all():
            print("[ERROR] logvar contiene NaNs")

        return x_hat, mu, logvar