import os
import numpy as np

import torch
import torch.nn.functional as F

import pandas as pd

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
        os.makedirs("samples", exist_ok=True)

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

        z = torch.randn(batch_size, self.LATENT_DIM, device=self.device)
        gen_imgs = self(z, genres)
        pred_fake = self.model.discriminator(gen_imgs, genres)
        g_loss = self.adversarial_loss(pred_fake, valid)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.log("g_loss", g_loss, prog_bar=True)
        self.train_step_metrics.append({"epoch": self.current_epoch, "batch": batch_idx, "g_loss": g_loss.item()})

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
            self.train_step_metrics[-1].update({"d_loss": d_loss.item()})

    def validation_step(self, batch, batch_idx):
        real_imgs, genres = batch
        batch_size = real_imgs.size(0)
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        z = torch.randn(batch_size, self.LATENT_DIM, device=self.device)
        gen_imgs = self(z, genres)
        pred_fake = self.model.discriminator(gen_imgs, genres)
        g_loss = self.adversarial_loss(pred_fake, valid)

        gen_imgs_detached = gen_imgs.detach()
        pred_real = self.model.discriminator(real_imgs, genres)
        pred_fake = self.model.discriminator(gen_imgs_detached, genres)
        d_loss_real = self.adversarial_loss(pred_real, valid)
        d_loss_fake = self.adversarial_loss(pred_fake, fake)
        d_loss = (d_loss_real + d_loss_fake) / 2

        self.log("val_g_loss", g_loss, prog_bar=True)
        self.log("val_d_loss", d_loss, prog_bar=True)
        self.val_step_metrics.append({"epoch": self.current_epoch, "batch": batch_idx, "val_g_loss": g_loss.item(), "val_d_loss": d_loss.item()})

    def on_train_epoch_end(self):
        if self.train_step_metrics:
            df = pd.DataFrame(self.train_step_metrics)
            df.to_csv("logs/csv/gan_train_step.csv", mode='a', header=not os.path.exists("logs/csv/gan_train_step.csv"), index=False)
            self.train_step_metrics.clear()

        if self.val_step_metrics:
            df_val = pd.DataFrame(self.val_step_metrics)
            df_val.to_csv("logs/csv/gan_val_step.csv", mode='a', header=not os.path.exists("logs/csv/gan_val_step.csv"), index=False)
            self.val_step_metrics.clear()
    
    def on_validation_epoch_end(self):
        if self.val_step_metrics:
            val_losses = [m["val_loss"] for m in self.val_step_metrics if "val_loss" in m]
            mean_val_loss = float(np.mean(val_losses))
            self.val_epoch_metrics.append({
                "epoch": self.current_epoch,
                "val_loss": mean_val_loss
            })
    def configure_optimizers(self):
        lr = 2e-4
        opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]
