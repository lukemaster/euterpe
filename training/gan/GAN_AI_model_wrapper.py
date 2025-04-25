
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
import json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from contextlib import redirect_stdout

from training.AI_model import AIModel
from training.config import Config

cfg = Config()

class GANAIModelWrapper(AIModel):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_name = "GAN"
        self.automatic_optimization = False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.base_log_dir = os.path.join("logs", f"{timestamp}_{self.model_name}_{cfg.KIND_OF_SPECTROGRAM}")
        self.csv_dir = os.path.join(self.base_log_dir, "csv")
        self.img_dir = os.path.join(self.base_log_dir, "img")
        self.ckpt_dir = os.path.join(self.base_log_dir, "checkpoints")
        self.samples_dir = os.path.join(self.base_log_dir, "samples")

        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        with open(os.path.join(self.base_log_dir, "config.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=4)

        summary_path = os.path.join(self.base_log_dir, "model_summary.txt")
        with open(summary_path, "w") as f:
            with redirect_stdout(f):
                print("=== Generator ===")
                print(self.model.generator)
                print("\n=== Discriminator ===")
                print(self.model.discriminator)

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
                z = torch.randn(
                    batch_size,
                    cfg.LATENT_DIM,
                    cfg.SPEC_ROWS // (2 ** (len(cfg.CNN) - 1)),
                    cfg.SPEC_COLS // (2 ** (len(cfg.CNN) - 1)),
                    device=self.device
                )
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

        z = torch.randn(
            batch_size,
            cfg.LATENT_DIM,
            cfg.SPEC_ROWS // (2 ** (len(cfg.CNN) - 1)),
            cfg.SPEC_COLS // (2 ** (len(cfg.CNN) - 1)),
            device=self.device
        )
        gen_imgs = self(z, genres)
        pred_fake = self.model.discriminator(gen_imgs, genres)
        g_loss = self.adversarial_loss(pred_fake, valid)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.log("g_loss", g_loss, prog_bar=True)
        self.train_step_metrics.append({"epoch": self.current_epoch, "batch": batch_idx, "g_loss": g_loss.item()})

        if not (self.current_epoch == 0 and batch_idx == 0):
            z = torch.randn(
                batch_size,
                cfg.LATENT_DIM,
                cfg.SPEC_ROWS // (2 ** (len(cfg.CNN) - 1)),
                cfg.SPEC_COLS // (2 ** (len(cfg.CNN) - 1)),
                device=self.device
            )
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

        # # === Inicialización a cero tras primera iteración
        # if self.current_epoch == 0 and batch_idx == 1:
        #     print("[INFO] Reinicializando pesos de ConvTranspose2d a cero tras primera iteración...")
        #     for module in self.model.generator.deconv:
        #         if isinstance(module, torch.nn.ConvTranspose2d):
        #             torch.nn.init.constant_(module.weight, 0.0)
        #             if module.bias is not None:
        #                 torch.nn.init.constant_(module.bias, 0.0)


        # ... dentro del método training_step (al final del todo)
        if batch_idx % 500 == 0:
            z_np = z[0, 0].detach().cpu().numpy()
            gen_np = gen_imgs[0, 0].detach().cpu().numpy()

            fig, axes = plt.subplots(2, 1, figsize=(10, 6))

            img1 = librosa.display.specshow(
                z_np,
                sr=cfg.SAMPLE_RATE,
                hop_length=cfg.HOP_LENGTH,
                x_axis="time",
                y_axis="linear",
                ax=axes[0]
            )
            axes[0].set_title(f"[Input Z] Epoch {self.current_epoch}, Batch {batch_idx}")
            fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

            img2 = librosa.display.specshow(
                gen_np,
                sr=cfg.SAMPLE_RATE,
                hop_length=cfg.HOP_LENGTH,
                x_axis="time",
                y_axis="linear",
                ax=axes[1]
            )
            axes[1].set_title(f"[Generated] Epoch {self.current_epoch}, Batch {batch_idx}")
            fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

            plt.tight_layout()
            plt.savefig(os.path.join(self.img_dir, f"spec_epoch{self.current_epoch}_batch{batch_idx}.png"))
            plt.close(fig)  # Cierra completamente la figura y libera memoria

            del fig, axes, img1, img2, z_np, gen_np  # Elimina referencias explícitas
            gc.collect()  # Fuerza liberación de memoria




    def validation_step(self, batch, batch_idx):
        real_imgs, genres = batch
        batch_size = real_imgs.size(0)
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        z = torch.randn(
            batch_size,
            cfg.LATENT_DIM,
            cfg.SPEC_ROWS // (2 ** (len(cfg.CNN) - 1)),
            cfg.SPEC_COLS // (2 ** (len(cfg.CNN) - 1)),
            device=self.device
        )
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
            for row in self.train_step_metrics:
                row["model_name"] = self.model_name
            df = pd.DataFrame(self.train_step_metrics)
            df.to_csv(os.path.join(self.csv_dir, "gan_train_step.csv"), mode='a', header=not os.path.exists(os.path.join(self.csv_dir, "gan_train_step.csv")), index=False)

            df_epoch = {
                "epoch": self.current_epoch,
                "g_loss": df["g_loss"].mean(),
                "model_name": self.model_name
            }
            if "d_loss" in df:
                df_epoch["d_loss"] = df["d_loss"].mean()

            self.train_epoch_metrics.append(df_epoch)
            pd.DataFrame(self.train_epoch_metrics).to_csv(os.path.join(self.csv_dir, "gan_train_epoch.csv"), index=False)

            self.train_step_metrics.clear()

        if self.val_step_metrics:
            for row in self.val_step_metrics:
                row["model_name"] = self.model_name
            df_val = pd.DataFrame(self.val_step_metrics)
            df_val.to_csv(os.path.join(self.csv_dir, "gan_val_step.csv"), mode='a', header=not os.path.exists(os.path.join(self.csv_dir, "gan_val_step.csv")), index=False)

            df_epoch_val = {
                "epoch": self.current_epoch,
                "model_name": self.model_name
            }
            if "val_g_loss" in df_val:
                df_epoch_val["val_g_loss"] = df_val["val_g_loss"].mean()
            if "val_d_loss" in df_val:
                df_epoch_val["val_d_loss"] = df_val["val_d_loss"].mean()
            self.val_epoch_metrics.append(df_epoch_val)
            pd.DataFrame(self.val_epoch_metrics).to_csv(os.path.join(self.csv_dir, "gan_val_epoch.csv"), index=False)

            self.val_step_metrics.clear()

        current_lr_g = self.trainer.optimizers[0].param_groups[0]["lr"]
        current_lr_d = self.trainer.optimizers[1].param_groups[0]["lr"]
        df_lr = pd.DataFrame([{
            "epoch": self.current_epoch,
            "lr_g": current_lr_g,
            "lr_d": current_lr_d,
            "model_name": self.model_name
        }])
        df_lr.to_csv(os.path.join(self.csv_dir, "gan_lr.csv"), mode='a', header=not os.path.exists(os.path.join(self.csv_dir, "gan_lr.csv")), index=False)

        if len(self.train_epoch_metrics) > 1:
            df_plot = pd.DataFrame(self.train_epoch_metrics)
            plt.figure()
            plt.plot(df_plot["epoch"], df_plot["g_loss"], label="g_loss")
            if "d_loss" in df_plot:
                plt.plot(df_plot["epoch"], df_plot["d_loss"], label="d_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("GAN losses")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.img_dir, f"gan_losses_epoch_{self.current_epoch}.jpg"))
            plt.close()

        torch.save(self.state_dict(), os.path.join(self.ckpt_dir, f"gan_last_{self.current_epoch}.pt"))

        if self.val_epoch_metrics:
            current_val_loss = self.val_epoch_metrics[-1].get("val_g_loss", None)
            if current_val_loss is not None and current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                torch.save(self.state_dict(), os.path.join(self.ckpt_dir, "gan_best.pt"))
                print(f"[INFO] New best GAN model saved (val_g_loss={current_val_loss:.4f})")

    
    def on_validation_epoch_end(self):
        if self.val_step_metrics:
            val_losses = [m["val_loss"] for m in self.val_step_metrics if "val_loss" in m]
            mean_val_loss = float(np.mean(val_losses))
            self.val_epoch_metrics.append({
                "epoch": self.current_epoch,
                "val_loss": mean_val_loss
            })

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.GAN_BETA_MIN, cfg.GAN_BETA_MAX))
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.GAN_BETA_MIN, cfg.GAN_BETA_MAX))
        return [opt_g, opt_d]
