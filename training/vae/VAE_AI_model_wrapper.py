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
import numpy as np

import torch
import torch.nn.functional as F


import pandas as pd
import matplotlib.pyplot as plt

from training.AI_model import AIModel
from training.config import Config

cfg = Config()

class VAEAIModelWrapper(AIModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean_x_buffer", torch.tensor(0.0))
        self.register_buffer("std_x_buffer", torch.tensor(1.0))

        os.makedirs("logs/csv", exist_ok=True)
        os.makedirs("logs/img", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
    
    def forward(self, x, genre):
        x = x[..., :cfg.SPEC_TIME_STEPS]

        x_hat, mu, logvar = self.model(x, genre)
        x_hat = x_hat[..., :cfg.SPEC_TIME_STEPS]
        if x_hat.shape[-1] != x.shape[-1]:
            min_width = min(x_hat.shape[-1], x.shape[-1])
            x = x[..., :min_width]
            x_hat = x_hat[..., :min_width]
        return x_hat, x, mu, logvar

    def compute_loss(self, x_hat, x, mu, logvar):
        # warmup = max(cfg.BETA_WARMUP_EPOCHS, 1)
        # step = self.current_epoch / warmup
        # beta = float(cfg.BETA_MAX / (1 + math.exp(-10 * (step - 0.5))))
        ##
        # step = self.global_step / self.total_steps
        # beta = float(cfg.BETA_MAX / (1 + math.exp(-10 * (step - 0.5))))
        ##
        # beta = min(beta, cfg.BETA_MAX)
        if self.global_step < self.total_steps * 0.3:
            beta = cfg.BETA_MAX * self.global_step / (self.total_steps * 0.3)
        else:
            beta = cfg.BETA_MAX
        ##
        # beta = 1
        

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = recon_loss + beta * kl_div
        return loss, recon_loss, kl_div, 1

    def training_step(self, batch, batch_idx):
        x, genre = batch
        x_hat, x, mu, logvar = self(x, genre)

        self.x_sum += x.sum()
        self.x_squared_sum += (x ** 2).sum()
        self.x_count += x.numel()

        for i in range(x.size(0)):
            self.update_genre_limits(genre[i].item(), x[i])

        loss, recon_loss, kl_div, beta = self.compute_loss(x_hat, x, mu, logvar)
        
        global_step = self.global_step

        if global_step % 500 == 0:
            print(f"[DEBUG] KL={kl_div:.4f} | Recon Loss={recon_loss:.4f} | mu mean={mu.mean().item():.4f} std={mu.std().item():.4f}")

            base_dir = "training_debug"
            recon_dir = os.path.join(base_dir, "reconstructions")
            mu_hist_dir = os.path.join(base_dir, "mu_histograms")
            logvar_hist_dir = os.path.join(base_dir, "logvar_histograms")

            os.makedirs(recon_dir, exist_ok=True)
            os.makedirs(mu_hist_dir, exist_ok=True)
            os.makedirs(logvar_hist_dir, exist_ok=True)

            spec_real = self.generate_spectrogram(x[0, 0])
            spec_recon = self.generate_spectrogram(x_hat[0, 0])

            spec_real = (spec_real + 1) / 2 * (cfg.DB_MAX - cfg.DB_MIN) + cfg.DB_MIN
            spec_recon = (spec_recon + 1) / 2 * (cfg.DB_MAX - cfg.DB_MIN) + cfg.DB_MIN

            _, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(spec_real, aspect='auto', origin='lower', cmap='magma')
            axs[0].set_title('Original')
            axs[1].imshow(spec_recon, aspect='auto', origin='lower', cmap='magma')
            axs[1].set_title('Reconstructed')
            for ax in axs:
                ax.set_xlabel('Time')
                ax.set_ylabel('Frequence')
            plt.tight_layout()

            fname = os.path.join(recon_dir, f"recon_step_{global_step}.jpg")
            plt.savefig(fname)
            plt.close()
            print(f"[DEBUG] Reconstruction saved in {fname}")

            mu_np = mu.detach().cpu().numpy().flatten()
            plt.figure(figsize=(6, 4))
            plt.hist(mu_np, bins=100, color="skyblue", edgecolor="black")
            plt.title(f"mu Histogram - step {global_step}")
            plt.xlabel("Value")
            plt.ylabel("Frequence")
            plt.tight_layout()
            mu_fname = os.path.join(mu_hist_dir, f"mu_hist_step_{global_step}.png")
            plt.savefig(mu_fname)
            plt.close()
            print(f"[DEBUG] mu histogram saved in {mu_fname}")

            logvar_np = logvar.detach().cpu().numpy().flatten()
            plt.figure(figsize=(6, 4))
            plt.hist(logvar_np, bins=100, color="salmon", edgecolor="black")
            plt.title(f"logvar histogram - step {global_step}")
            plt.xlabel("Value")
            plt.ylabel("Frequence")
            plt.tight_layout()
            logvar_fname = os.path.join(logvar_hist_dir, f"logvar_hist_step_{global_step}.png")
            plt.savefig(logvar_fname)
            plt.close()
            print(f"[DEBUG] logvar  histogram saved in {logvar_fname}")

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
            print(f"[INFO] Epoc {self.current_epoch} completed — mean_x={self.mean_x:.4f}, std_x={self.std_x:.4f}")
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
            plt.title("x mean and std")
            plt.savefig(f"logs/img/stats_epoch_{self.current_epoch}.jpg")
            plt.close()

        torch.save({"model": self.state_dict(), "extra": extra}, f'''checkpoints/vae_last_{self.current_epoch}.pt''')

        if self.val_epoch_metrics:
            current_val_loss = self.val_epoch_metrics[-1]["val_loss"]
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                torch.save({"model": self.state_dict(), "extra": extra}, "checkpoints/vae_best.pt")
                torch.save(self.state_dict(), f"checkpoints/vae_best.pt")
                print(f"[INFO] New best model saved (val_loss={current_val_loss:.4f})")

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
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=cfg.LR_SCHEDULER_PATIENTE, factor=cfg.LR_SCHEDULER_FACTOR, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }