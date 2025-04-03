import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from training.config import Config

cfg = Config()

class VAE(nn.Module):
    NUM_GENRES = cfg.NUM_GENRES
    GENRE_EMBEDDING_DIM = (2 ** math.ceil(math.log2(NUM_GENRES)))# * 2 maybe an improvement or not

    class VAE_Encoder(nn.Module):
        def __init__(self, cfg: Config):
            super().__init__()
            self.cfg = cfg
            self.input_dim = 513 * 344
            self.cond_dim = 5  # Puede parametrizarse si se desea
            self.latent_dim = cfg.LATENT_DIM

            self.genre_emb = nn.Embedding(cfg.NUM_GENRES, self.cond_dim)

            self.fc1 = nn.Linear(self.input_dim + self.cond_dim, 512)
            self.fc_mu = nn.Linear(512, self.latent_dim)
            self.fc_logvar = nn.Linear(512, self.latent_dim)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor, c: torch.Tensor):
            x = x.reshape(x.size(0), -1)  # [B, N]
            c = c.view(x.size(0))         # [B]
            c = self.genre_emb(c)         # [B, emb_dim]
            x_cond = torch.cat([x, c], dim=1)  # [B, N + emb_dim]

            h = F.relu(self.fc1(x_cond))
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)

            return mu, logvar


    class VAE_Decoder(nn.Module):
        def __init__(self, cfg: Config):
            super().__init__()
            self.cfg = cfg
            self.output_shape = (1, 513, 344)
            self.output_dim = 513 * 344
            self.cond_dim = 5
            self.latent_dim = cfg.LATENT_DIM

            self.genre_emb = nn.Embedding(cfg.NUM_GENRES, self.cond_dim)

            self.fc = nn.Linear(self.latent_dim + self.cond_dim, 512 * 344)
            self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
            self.fc_out = nn.Linear(512, 513)

        def forward(self, z: torch.Tensor, c: torch.Tensor):
            B = z.size(0)
            c = c.reshape(B)
            c = self.genre_emb(c)  # [B, cond_dim]

            z_cond = torch.cat([z, c], dim=1)  # [B, LATENT_DIM + cond_dim]
            x = self.fc(z_cond)  # [B, 512 * 344]
            x = x.view(B, 344, 512)  # [B, T=344, F=512]
            x, _ = self.lstm(x)  # [B, 344, 512]
            x = self.fc_out(x)  # [B, 344, 513]
            x = x.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 513, 344]
            x = torch.tanh(x)
            return x

            
    def __init__(self):
        super().__init__()
        input_shape = (1, cfg.SPEC_ROWS, cfg.SPEC_TIME_STEPS)
        cond_dim=cfg.NUM_GENRES
        latent_dim=cfg.LATENT_DIM
        hidden_dim=1024

        self.LATENT_DIM = cfg.LATENT_DIM
        self.SPEC_TIME_STEPS = cfg.SPEC_TIME_STEPS
        self.encoder = self.VAE_Encoder(cfg)
        self.decoder = self.VAE_Decoder(cfg)

        # self.apply(self.init_weights)
        
    # def init_weights(_,m):
    #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
    #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #     elif isinstance(m, nn.Embedding):
    #         nn.init.uniform_(m.weight, -0.1, 0.1)
    #     elif isinstance(m, nn.LSTM):
    #         for name, param in m.named_parameters():
    #             if 'weight' in name:
    #                 nn.init.xavier_uniform_(param)
    #             elif 'bias' in name:
    #                 nn.init.zeros_(param)


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar