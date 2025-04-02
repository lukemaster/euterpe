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

from training.AI_model import AIModel
from training.config import Config

cfg = Config()

class VAEAIModelWrapper(AIModel):
    BETA_MAX = cfg.BETA_MAX
    BETA_WARMUP_EPOCHS = cfg.BETA_WARMUP_EPOCHS
    LATENT_DIM = cfg.LATENT_DIM
    SAMPLE_RATE = cfg.SAMPLE_RATE
    N_FFT = cfg.N_FFT
    HOP_LENGTH = cfg.HOP_LENGTH
    SPEC_TIME_STEPS = cfg.SPEC_TIME_STEPS
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean_x_buffer", torch.tensor(0.0))
        self.register_buffer("std_x_buffer", torch.tensor(1.0))

        os.makedirs("logs/csv", exist_ok=True)
        os.makedirs("logs/img", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
    
    def forward(self, x, genre):
        x = x[..., :self.SPEC_TIME_STEPS]
        x_hat, mu, logvar = self.model(x, genre)
        x_hat = x_hat[..., :self.SPEC_TIME_STEPS]
        if x_hat.shape[-1] != x.shape[-1]:
            min_width = min(x_hat.shape[-1], x.shape[-1])
            x = x[..., :min_width]
            x_hat = x_hat[..., :min_width]
        return x_hat, x, mu, logvar

    def compute_loss(self, x_hat, x, mu, logvar):
        warmup = max(self.BETA_WARMUP_EPOCHS, 1)
        step = self.current_epoch / warmup
        beta = float(self.BETA_MAX / (1 + math.exp(-10 * (step - 0.5))))
        beta = min(beta, self.BETA_MAX)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = recon_loss + beta * kl_div
        return loss, recon_loss, kl_div, beta

    def training_step(self, batch, batch_idx):
        x, genre = batch
        x_hat, x, mu, logvar = self(x, genre)

        self.x_sum += x.sum()
        self.x_squared_sum += (x ** 2).sum()
        self.x_count += x.numel()

        for i in range(x.size(0)):
            self.update_genre_limits(genre[i].item(), x[i])

        loss, recon_loss, kl_div, beta = self.compute_loss(x_hat, x, mu, logvar)

        self.log("loss", loss, prog_bar=True)
        self.log("recon_loss", recon_loss, prog_bar=True)
        self.log("kl_div", kl_div, prog_bar=True)
        self.log("beta", beta, prog_bar=True)

        self.train_step_metrics.append({
            "epoch": self.current_epoch,
            "batch": batch_idx,
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_div": kl_div.item(),
            "beta": beta
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, genre = batch
        x_hat, x, mu, logvar = self(x, genre)
        loss, recon_loss, kl_div, beta = self.compute_loss(x_hat, x, mu, logvar)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_kl_div", kl_div, prog_bar=True)

        self.val_step_metrics.append({
            "epoch": self.current_epoch,
            "batch": batch_idx,
            "val_loss": loss.item(),
            "val_recon_loss": recon_loss.item(),
            "val_kl_div": kl_div.item(),
            "beta": beta
        })
        return loss
    
    def on_train_epoch_end(self):
        if self.x_count > 0:
            mean_x = self.x_sum / self.x_count
            std_x = torch.sqrt(self.x_squared_sum / self.x_count - mean_x ** 2)
            self.mean_x = mean_x.item()
            self.std_x = std_x.item()
            self.log("x_mean_epoch", mean_x, prog_bar=True)
            self.log("x_std_epoch", std_x, prog_bar=True)
            print(f"[INFO] Época {self.current_epoch} completada — mean_x={self.mean_x:.4f}, std_x={self.std_x:.4f}")
            self.mean_x_buffer.copy_(mean_x)
            self.std_x_buffer.copy_(std_x)

        self.x_sum.zero_()
        self.x_squared_sum.zero_()
        self.x_count.zero_()

        if self.train_step_metrics:
            df_train = pd.DataFrame(self.train_step_metrics)
            df_train.to_csv("logs/csv/metrics_train_step.csv", mode='a', header=not os.path.exists("logs/csv/metrics_train_step.csv"), index=False)
            self.train_step_metrics.clear()

        if self.val_step_metrics:
            df_val = pd.DataFrame(self.val_step_metrics)
            df_val.to_csv("logs/csv/metrics_val_step.csv", mode='a', header=not os.path.exists("logs/csv/metrics_val_step.csv"), index=False)
            self.val_step_metrics.clear()

        epoch_metrics = {
            "epoch": self.current_epoch,
            "x_mean": self.mean_x,
            "x_std": self.std_x
        }
        self.train_epoch_metrics.append(epoch_metrics)
        df_epoch = pd.DataFrame(self.train_epoch_metrics)
        df_epoch.to_csv("logs/csv/metrics_train_epoch.csv", index=False)

        if self.val_epoch_metrics:
            df_val_epoch = pd.DataFrame(self.val_epoch_metrics)
            df_val_epoch.to_csv("logs/csv/metrics_val_epoch.csv", index=False)

        extra = {
            "mean_x": self.mean_x,
            "std_x": self.std_x
        }

        if len(df_epoch) > 1:
            plt.figure()
            plt.plot(df_epoch["epoch"], df_epoch["x_mean"], label="x_mean")
            plt.plot(df_epoch["epoch"], df_epoch["x_std"], label="x_std")
            plt.xlabel("Epoch")
            plt.ylabel("Valor")
            plt.legend()
            plt.title("Media y desviación estándar de x")
            plt.savefig(f"logs/img/stats_epoch_{self.current_epoch}.jpg")
            plt.close()

        torch.save({"model": self.state_dict(), "extra": extra}, f'''checkpoints/vae_last_{self.current_epoch}.pt''')

        if self.val_epoch_metrics:
            current_val_loss = self.val_epoch_metrics[-1]["val_loss"]
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                torch.save({"model": self.state_dict(), "extra": extra}, "checkpoints/vae_best.pt")
                torch.save(self.state_dict(), f"checkpoints/vae_best.pt")
                print(f"[INFO] Nuevo mejor modelo guardado (val_loss={current_val_loss:.4f})")

        self.save_genre_limits()

    def on_validation_epoch_end(self):
        if self.val_step_metrics:
            val_g_losses = [m["val_g_loss"] for m in self.val_step_metrics if "val_g_loss" in m]
            val_d_losses = [m["val_d_loss"] for m in self.val_step_metrics if "val_d_loss" in m]
            if val_g_losses and val_d_losses:
                self.val_epoch_metrics.append({
                    "epoch": self.current_epoch,
                    "val_g_loss": float(np.mean(val_g_losses)),
                    "val_d_loss": float(np.mean(val_d_losses))
                })
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }