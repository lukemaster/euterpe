import os
import math


import torch
import torch.nn as nn
import torch.nn.functional as F


from training.config import Config

cfg = Config()

class VAE(nn.Module):
    NUM_GENRES = cfg.NUM_GENRES
    GENRE_EMBEDDING_DIM = (2 ** math.ceil(math.log2(NUM_GENRES)))# * 2 maybe an improvement or not

    class VAE_Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.debug = cfg.debug
            in_channels = cfg.CNN[0]
            layers = []
            current_channels = in_channels

            for out_channels in cfg.CNN[1:]:
                layers.append(nn.Conv2d(current_channels, out_channels,
                                        kernel_size=cfg.CNN_KERNEL,
                                        stride=cfg.CNN_STRIDE,
                                        padding=cfg.CNN_PADDING))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                current_channels = out_channels

            self.encoder = nn.Sequential(*layers)

            self.genre_embedding = nn.Embedding(cfg.NUM_GENRES, cfg.GENRE_EMBED_DIM)

            self.conv_mu = nn.Conv2d(current_channels + cfg.GENRE_EMBED_DIM,
                                    cfg.LATENT_CHANNELS, kernel_size=1)
            self.conv_logvar = nn.Conv2d(current_channels + cfg.GENRE_EMBED_DIM,
                                        cfg.LATENT_CHANNELS, kernel_size=1)

        def forward(self, x, genre):
            if self.debug:
                print(f"[ENCODER] Input x: {x.shape}")  # e.g. (B, 1, 513, 344)

            x = self.encoder(x)

            if self.debug:
                print(f"[ENCODER] After CNN stack: {x.shape}")

            genre_emb = self.genre_embedding(genre).unsqueeze(-1).unsqueeze(-1)
            genre_emb = genre_emb.expand(-1, -1, x.shape[2], x.shape[3])

            x = torch.cat([x, genre_emb], dim=1)

            if self.debug:
                print(f"[ENCODER] After genre concat: {x.shape}")

            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            # logvar = torch.clamp(logvar, min=-10, max=10)


            if self.debug:
                print(f"[ENCODER] mu: {mu.shape}, logvar: {logvar.shape}")

            return mu, logvar


    class VAE_Decoder(nn.Module):
        def __init__(self, flattened_shape: tuple):
            super().__init__()
            self.debug = cfg.debug
            self.latent_channels, self.h, self.w = flattened_shape
            self.seq_len = self.w
            self.input_size = self.latent_channels * self.h
            self.hidden_size = cfg.LSTM_HIDDEN_SIZE
            self.genre_embedding = nn.Embedding(cfg.NUM_GENRES, cfg.GENRE_EMBED_DIM)

            self.lstm = nn.LSTM(input_size=self.input_size + cfg.GENRE_EMBED_DIM,
                                hidden_size=self.hidden_size,
                                num_layers=cfg.LSTM_NUM_LAYERS,
                                batch_first=True)

            # Cada paso temporal reconstruye un bloque plano que luego se reagrupa
            self.fc_out = nn.Linear(self.hidden_size, self.latent_channels * self.h)

            # Deconvoluciones para ampliar desde volumen reducido a espectrograma completo
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(self.latent_channels, cfg.CNN[-1],
                                kernel_size=cfg.CNN_KERNEL,
                                stride=cfg.CNN_STRIDE,
                                padding=cfg.CNN_PADDING),
                nn.ReLU(),

                nn.ConvTranspose2d(cfg.CNN[-1], cfg.CNN[-2],
                                kernel_size=cfg.CNN_KERNEL,
                                stride=cfg.CNN_STRIDE,
                                padding=cfg.CNN_PADDING),
                nn.ReLU(),

                nn.ConvTranspose2d(cfg.CNN[-2], cfg.CNN[0],
                                kernel_size=cfg.CNN_KERNEL,
                                stride=cfg.CNN_STRIDE,
                                padding=cfg.CNN_PADDING,
                                output_padding=(1, 1))  # Corrige desfase en frecuencia
            )

        def forward(self, z, genre):
            if self.debug:
                print(f"[DECODER] Input z: {z.shape}")  # (B, C, H, W)

            z_seq = z.view(z.size(0), self.latent_channels * self.h, self.w).permute(0, 2, 1)

            if self.debug:
                print(f"[DECODER] z_seq for LSTM: {z_seq.shape}")  # (B, T, F)

            genre_emb = self.genre_embedding(genre).unsqueeze(1)
            genre_seq = genre_emb.expand(-1, z_seq.size(1), -1)
            z_input = torch.cat([z_seq, genre_seq], dim=-1)

            if self.debug:
                print(f"[DECODER] z_input to LSTM: {z_input.shape}")

            output, _ = self.lstm(z_input)

            if self.debug:
                print(f"[DECODER] LSTM output: {output.shape}")

            output = self.fc_out(output)

            if self.debug:
                print(f"[DECODER] After fc_out: {output.shape}")

            volume = output.permute(0, 2, 1).view(z.size(0), self.latent_channels, self.h, self.w)

            if self.debug:
                print(f"[DECODER] Volume for deconv: {volume.shape}")

            recon = self.deconv(volume)

            if self.debug:
                print(f"[DECODER] Final recon: {recon.shape}")

            return recon


            
    def __init__(self):
        super().__init__()
        self.encoder = self.VAE_Encoder()

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.CNN[0], cfg.SPEC_ROWS, cfg.SPEC_COLS)
            dummy_genre = torch.zeros(1, dtype=torch.long)
            mu, _ = self.encoder(dummy, dummy_genre)
            self.flattened_shape = mu.shape[1:]  # (C, H, W)

        self.decoder = self.VAE_Decoder(self.flattened_shape)

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
        mu, logvar = self.encoder(x, genre)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, genre)
        return x_hat, mu, logvar