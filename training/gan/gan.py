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

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from training.config import Config

cfg = Config()

class GAN(nn.Module):
    class GAN_Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_channels, self.h, self.w = (
                cfg.LATENT_DIM,
                cfg.SPEC_ROWS // (2 ** (len(cfg.CNN) - 1)),
                cfg.SPEC_COLS // (2 ** (len(cfg.CNN) - 1))
            )
            self.seq_len = self.w
            self.token_dim = self.latent_channels * self.h

            self.genre_embedding = nn.Embedding(cfg.NUM_GENRES, cfg.GENRE_EMBED_DIM)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.token_dim + cfg.GENRE_EMBED_DIM,
                nhead=cfg.GEN_NUM_HEADS,
                dim_feedforward=cfg.GEN_MODEL_DIM * 2,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.GEN_NUM_LAYERS)

            self.fc_out = nn.Linear(self.token_dim + cfg.GENRE_EMBED_DIM, self.token_dim)

            if cfg.KIND_OF_SPECTROGRAM == 'MEL':
                deconv_kernel = 4
                deconv_stride = 2
                deconv_padding = 1
            else:
                deconv_kernel = cfg.CNN_KERNEL
                deconv_stride = cfg.CNN_STRIDE
                deconv_padding = cfg.CNN_PADDING


            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(self.latent_channels, cfg.CNN[-1],
                    kernel_size=(deconv_kernel, 1),    # kernel alto, estrecho
                    stride=(deconv_stride, 1),
                    padding=(deconv_padding, 0),
                    # output_padding=(0, 0)
                ),
                nn.LeakyReLU(0.2),

                nn.ConvTranspose2d(cfg.CNN[-1], cfg.CNN[-2],
                    kernel_size=(deconv_kernel, 1),
                    stride=(deconv_stride, 1),
                    padding=(deconv_padding, 0),
                    # output_padding=(0, 0)
                ),
                nn.LeakyReLU(0.2),

                nn.ConvTranspose2d(cfg.CNN[-2], cfg.CNN[0],
                    kernel_size=(deconv_kernel, 1),
                    stride=(deconv_stride, 1),
                    padding=(deconv_padding, 0),
                    # output_padding=(0, 0)
                ),
                nn.Tanh()
            )

        def forward(self, z, genre):
            B = z.size(0)

            z_seq = z.view(B, self.token_dim, self.seq_len).permute(0, 2, 1)

            genre_emb = self.genre_embedding(genre).unsqueeze(1)
            genre_seq = genre_emb.expand(-1, self.seq_len, -1)

            z_input = torch.cat([z_seq, genre_seq], dim=-1)

            x = self.transformer(z_input)
            x = self.fc_out(x)

            volume = x.permute(0, 2, 1).view(B, self.latent_channels, self.h, self.w)

            recon = self.deconv(volume)
            return recon

    class GAN_Discriminator(nn.Module):
        def __init__(self):
            super().__init__()

            self.genre_embedding = nn.Embedding(cfg.NUM_GENRES, cfg.GENRE_EMBED_DIM)

            self.conv1 = nn.Conv2d(1 + cfg.GENRE_EMBED_DIM, 64, kernel_size=(cfg.CNN_KERNEL,1), padding=(cfg.CNN_PADDING, 0))
            self.bn1 = nn.BatchNorm2d(64)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=(cfg.CNN_KERNEL,1), padding=(cfg.CNN_PADDING, 0))
            self.bn2 = nn.BatchNorm2d(128)

            self.conv3 = nn.Conv2d(128, 256, kernel_size=(cfg.CNN_KERNEL,1), padding=(cfg.CNN_PADDING, 0))
            self.bn3 = nn.BatchNorm2d(256)

            self.pool = nn.MaxPool2d(kernel_size=2)
            self.flatten = nn.Flatten()
            
            self._classifier = None
            self._last_shape = None

        def forward(self, x: torch.Tensor, genre: torch.Tensor) -> torch.Tensor:
            genre_emb = self.genre_embedding(genre).view(genre.size(0), cfg.GENRE_EMBED_DIM, 1, 1)
            genre_emb = genre_emb.expand(-1, -1, x.size(2), x.size(3))

            x = torch.cat([x, genre_emb], dim=1)

            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))

            x = self.flatten(x)

            if self._classifier is None or x.shape[1] != self._last_shape:
                self._last_shape = x.shape[1]
                self._classifier = nn.Linear(x.shape[1], 1).to(x.device)

            return self._classifier(x)

    class PretrainingGAN(pl.LightningModule):
        def __init__(self,gan):
            super().__init__()
            self.generator = gan.generator
            self.discriminator = gan.discriminator
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            real_specs, genres = batch
            batch_size = real_specs.size(0)

            opt_g, opt_d = self.optimizers()

            # Generator pretraining
            if self.current_epoch < cfg.GAN_PRETRAIN_EPOCHS_G:
                z = torch.randn(
                    batch_size,
                    cfg.LATENT_DIM,
                    cfg.SPEC_ROWS // (2 ** (len(cfg.CNN) - 1)),
                    cfg.SPEC_COLS // (2 ** (len(cfg.CNN) - 1)),
                    device=self.device
                )
                gen_specs = self.generator(z, genres)
                real_specs = real_specs[..., :cfg.SPEC_TIME_STEPS]
                g_loss = F.mse_loss(gen_specs, real_specs)
                opt_g.zero_grad()
                self.manual_backward(g_loss)
                opt_g.step()
                self.log("pretrain_g_loss", g_loss, prog_bar=True)

            # Discriminator pretraining
            if self.current_epoch < cfg.GAN_PRETRAIN_EPOCHS_D:
                valid = torch.ones(batch_size, 1, device=self.device)
                preds = self.discriminator(real_specs, genres)
                d_loss = F.binary_cross_entropy_with_logits(preds, valid)
                opt_d.zero_grad()
                self.manual_backward(d_loss)
                opt_d.step()
                self.log("pretrain_d_loss", d_loss, prog_bar=True)

        def configure_optimizers(self):
            opt_g = torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
            opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
            return [opt_g, opt_d]

        def configure_optimizers(self):
            opt_g = torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
            opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
            return [opt_g, opt_d]

    def __init__(self):
        super().__init__()
        cfg.SPEC_TIME_STEPS = cfg.SPEC_TIME_STEPS
        self.automatic_optimization = False
        self.generator = self.GAN_Generator()
        self.discriminator = self.GAN_Discriminator()
        self.pretrainModule = self.PretrainingGAN(self)

    def forward(self, z, genre):
        return self.generator(z, genre)

    def adversarial_loss(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(y_pred, y_true)