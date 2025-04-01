import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


from dotenv import load_dotenv
load_dotenv('./VIU/09MIAR/euterpe/.env')


class GAN(nn.Module):
    NUM_GENRES = int(os.environ.get('NUM_GENRES'))
    GENRE_EMBEDDING_DIM = (2 ** math.ceil(math.log2(NUM_GENRES)))# * 2 maybe an improvement or not

    class GAN_Generator(nn.Module):

        def __init__(self, gan):
            super().__init__()
            self.NUM_GENRES = gan.NUM_GENRES
            self.LATENT_DIM = gan.LATENT_DIM
            self.GENRE_EMBEDDING_DIM = gan.GENRE_EMBEDDING_DIM
            self.SPEC_TIME_STEPS = gan.SPEC_TIME_STEPS
            self.GEN_MODEL_DIM = int(os.environ.get('GEN_MODEL_DIM'))
            self.GEN_NUM_LAYERS = int(os.environ.get('GEN_NUM_LAYERS'))
            self.GEN_NUM_HEADS = int(os.environ.get('GEN_NUM_HEADS'))
            self.NUM_MELS = int(os.environ["NUM_MELS"])

            self.genre_embedding = nn.Embedding(self.NUM_GENRES, self.GENRE_EMBEDDING_DIM)
            self.input_dim = self.LATENT_DIM + self.GENRE_EMBEDDING_DIM
            self.linear_in = nn.Linear(self.input_dim, self.SPEC_TIME_STEPS * self.GEN_MODEL_DIM)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.GEN_MODEL_DIM,
                nhead=self.GEN_NUM_HEADS,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.GEN_NUM_LAYERS)

            self.to_mel = nn.Sequential(
                nn.Conv1d(self.GEN_MODEL_DIM, self.GEN_MODEL_DIM // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(self.GEN_MODEL_DIM // 2, self.NUM_MELS, kernel_size=3, padding=1),
                nn.Tanh()
            )

        def forward(self, z: torch.Tensor, genre: torch.Tensor) -> torch.Tensor:
            genre_emb = self.genre_embedding(genre)
            x = torch.cat([z, genre_emb], dim=1)

            x = self.linear_in(x)
            x = x.view(x.size(0), self.SPEC_TIME_STEPS, self.GEN_MODEL_DIM)
            x = self.transformer(x)
            x = x.permute(0, 2, 1)
            x = self.to_mel(x)
            x = x.unsqueeze(1)
            return x[..., :self.SPEC_TIME_STEPS]

    class GAN_Discriminator(nn.Module):
        def __init__(self, gan):
            super().__init__()

            self.NUM_GENRES = gan.NUM_GENRES
            self.GENRE_EMBEDDING_DIM = gan.GENRE_EMBEDDING_DIM

            self.genre_embedding = nn.Embedding(self.NUM_GENRES, self.GENRE_EMBEDDING_DIM)

            self.conv1 = nn.Conv2d(1 + self.GENRE_EMBEDDING_DIM, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)

            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)

            self.pool = nn.MaxPool2d(kernel_size=2)

            self.flatten_shapes = {}
            self.final_dim = None
            self.fc = None

        def forward(self, x: torch.Tensor, genre: torch.Tensor) -> torch.Tensor:
            genre_emb = self.genre_embedding(genre).view(genre.size(0), self.GENRE_EMBEDDING_DIM, 1, 1)
            genre_emb = genre_emb.expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, genre_emb], dim=1)

            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))

            x = x.view(x.size(0), -1)

            if self.fc is None or self.final_dim != x.size(1):
                self.final_dim = x.size(1)
                self.fc = nn.Linear(self.final_dim, 1).to(x.device)

            return self.fc(x)

    class PretrainingGAN(pl.LightningModule):
        def __init__(self, gan):
            super().__init__()
            self.GAN_PRETRAIN_EPOCHS_G = int(os.environ.get('GAN_PRETRAIN_EPOCHS_G'))
            self.GAN_PRETRAIN_EPOCHS_D = int(os.environ.get('GAN_PRETRAIN_EPOCHS_D'))                
            self.LATENT_DIM = gan.LATENT_DIM
            self.SPEC_TIME_STEPS = gan.SPEC_TIME_STEPS

            self.generator = gan.generator
            self.discriminator = gan.discriminator
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            real_specs, genres = batch
            batch_size = real_specs.size(0)

            opt_g, opt_d = self.optimizers()

            # Preentrenamiento del generador
            if self.current_epoch < self.GAN_PRETRAIN_EPOCHS_G:
                z = torch.randn(batch_size, self.LATENT_DIM, device=self.device)
                gen_specs = self.generator(z, genres)
                real_specs = real_specs[..., :self.SPEC_TIME_STEPS]  # Recorte forzado
                g_loss = F.mse_loss(gen_specs, real_specs)
                opt_g.zero_grad()
                self.manual_backward(g_loss)
                opt_g.step()
                self.log("pretrain_g_loss", g_loss, prog_bar=True)

            # Preentrenamiento del discriminador
            if self.current_epoch < self.GAN_PRETRAIN_EPOCHS_D:
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
        self.LATENT_DIM = int(os.environ.get('LATENT_DIM'))
        self.SPEC_TIME_STEPS = int((int(os.environ["SAMPLE_RATE"]) * int(os.environ.get('SEGMENT_DURATION'))) / int(os.environ["HOP_LENGTH"]))
        self.automatic_optimization = False
        self.generator = self.GAN_Generator(self)
        self.discriminator = self.GAN_Discriminator(self)
        self.pretrainModule = self.PretrainingGAN(self)

    def forward(self, z, genre):
        return self.generator(z, genre)

    def adversarial_loss(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(y_pred, y_true)