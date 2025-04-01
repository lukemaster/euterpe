import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


from dotenv import load_dotenv
from training.AI_model import AIModel
load_dotenv('./VIU/09MIAR/euterpe/.env')


class GANAIModelWrapper(AIModel):
    LATENT_DIM = int(os.environ["LATENT_DIM"])
    SAMPLE_RATE = int(os.environ["SAMPLE_RATE"])
    HOP_LENGTH = int(os.environ["HOP_LENGTH"])
    SPEC_TIME_STEPS = int((SAMPLE_RATE * int(os.environ.get('SEGMENT_DURATION'))) / HOP_LENGTH)
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.automatic_optimization = False

    def forward(self, z, genre):
        return self.model.generator(z, genre)

    def adversarial_loss(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        real_imgs, genres = batch
        batch_size = real_imgs.size(0)
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        opt_g, opt_d = self.optimizers()

        # ============================================================
        # ENTRENAMIENTO DEL DISCRIMINADOR (se ejecuta primero en la 1ª iteración)
        # ============================================================
        if self.current_epoch == 0 and batch_idx == 0:
            with torch.no_grad():
                z = torch.randn(batch_size, self.LATENT_DIM, device=self.device)
                gen_imgs = self(z, genres).detach()

            pred_real = self.model.discriminator(real_imgs, genres)
            pred_fake = self.model.discriminator(gen_imgs, genres)

            d_loss_real = self.adversarial_loss(pred_real, valid)
            d_loss_fake = self.adversarial_loss(pred_fake, fake)
            d_loss = (d_loss_real + d_loss_fake) / 2

            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()
            self.log("d_loss", d_loss, prog_bar=True)

        # ============================================================
        # ENTRENAMIENTO DEL GENERADOR
        # ============================================================
        z = torch.randn(batch_size, self.LATENT_DIM, device=self.device)
        gen_imgs = self(z, genres)
        pred_fake = self.model.discriminator(gen_imgs, genres)
        g_loss = self.adversarial_loss(pred_fake, valid)

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.log("g_loss", g_loss, prog_bar=True)

        # ============================================================
        # ENTRENAMIENTO DEL DISCRIMINADOR (en pasos posteriores normales)
        # ============================================================
        if not (self.current_epoch == 0 and batch_idx == 0):
            z = torch.randn(batch_size, self.LATENT_DIM, device=self.device)
            gen_imgs = self(z, genres).detach()

            pred_real = self.model.discriminator(real_imgs, genres)
            pred_fake = self.model.discriminator(gen_imgs, genres)

            d_loss_real = self.adversarial_loss(pred_real, valid)
            d_loss_fake = self.adversarial_loss(pred_fake, fake)
            d_loss = (d_loss_real + d_loss_fake) / 2

            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()
            self.log("d_loss", d_loss, prog_bar=True)

    def on_train_epoch_end(self):#TODO
        pass  # desactivado temporalmente por memoria

    def configure_optimizers(self):
        lr = 2e-4
        opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]
