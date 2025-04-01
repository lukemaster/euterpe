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

class AIModel(pl.LightningModule):
    BETA_MAX = float(os.environ["BETA_MAX"])
    BETA_WARMUP_EPOCHS = int(os.environ["BETA_WARMUP_EPOCHS"])
    LATENT_DIM = int(os.environ["LATENT_DIM"])
    SAMPLE_RATE = int(os.environ["SAMPLE_RATE"])
    N_FFT = int(os.environ["N_FFT"])
    HOP_LENGTH = int(os.environ["HOP_LENGTH"])
    NUM_MELS = int(os.environ["NUM_MELS"])
    SPEC_TIME_STEPS = int((SAMPLE_RATE * int(os.environ.get('SEGMENT_DURATION'))) / HOP_LENGTH)
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("x_sum", torch.tensor(0.0))
        self.register_buffer("x_squared_sum", torch.tensor(0.0))
        self.register_buffer("x_count", torch.tensor(0))
        self.std_x = 0
        self.mean_x = 0
        self.best_val_loss = float("inf")

        self.genre_db_limits_path = "logs/genre_db_limits.json"
        self.genre_db_limits = self.load_or_init_genre_limits()

        self.train_step_metrics = []
        self.val_step_metrics = []
        self.train_epoch_metrics = []
        self.val_epoch_metrics = []

        self.register_buffer("mean_x_buffer", torch.tensor(0.0))
        self.register_buffer("std_x_buffer", torch.tensor(1.0))

        os.makedirs("logs/csv", exist_ok=True)
        os.makedirs("logs/img", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def load_or_init_genre_limits(self):
        if os.path.exists(self.genre_db_limits_path):
            with open(self.genre_db_limits_path, "r") as f:
                return json.load(f)
        return {str(i): [100.0, -100.0] for i in range(int(os.environ.get('NUM_GENRES')))}

    def update_genre_limits(self, genre_id, spec):
        spec_db = 20 * torch.log10(torch.clamp(spec, min=1e-5)).cpu().numpy()
        min_db = float(np.min(spec_db))
        max_db = float(np.max(spec_db))
        g = str(genre_id)
        if g not in self.genre_db_limits:
            self.genre_db_limits[g] = [min_db, max_db]
        else:
            current_min, current_max = self.genre_db_limits[g]
            self.genre_db_limits[g][0] = min(current_min, min_db)
            self.genre_db_limits[g][1] = max(current_max, max_db)

    def save_genre_limits(self):
        with open(self.genre_db_limits_path, "w") as f:
            json.dump(self.genre_db_limits, f, indent=2)

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
    
    def on_validation_epoch_end(self):
        if self.val_step_metrics:
            val_losses = [m["val_loss"] for m in self.val_step_metrics if "val_loss" in m]
            mean_val_loss = float(np.mean(val_losses))
            self.val_epoch_metrics.append({
                "epoch": self.current_epoch,
                "val_loss": mean_val_loss
            })

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

    def generate_audio_from_noise(self, genre_id: int, output_path: str = "generated.wav"):
        
        device = self.device
        self.eval()
        vae = self.model
        KIND_OF_SPECTROGRAM = os.getenv("KIND_OF_SPECTROGRAM", "MEL").upper()


        with open(self.model.genre_db_limits_path, "r") as f:
            genre_db_limits = json.load(f)
        db_min, db_max = genre_db_limits[str(genre_id)]

        num_segments = int(int(os.environ.get('TOTAL_DURATION')) / int(os.environ.get('SEGMENT_DURATION')))
        zs = [torch.randn(1, self.LATENT_DIM).to(device) for _ in range(num_segments)]
        genre = torch.tensor([genre_id], dtype=torch.long).to(device)

        
        with torch.no_grad():
            decoded_specs = []
            for z in zs:
                spec = vae.decoder(z, genre).squeeze(0).cpu()
                decoded_specs.append(spec)

        n = 20
        for i in range(len(decoded_specs)):
            if decoded_specs[i].shape[-1] >= 3:
                decoded_specs[i][..., :n] = decoded_specs[i][..., n].unsqueeze(-1).expand_as(decoded_specs[i][..., :n])
                decoded_specs[i][..., -n:] = decoded_specs[i][..., -n-1].unsqueeze(-1).expand_as(decoded_specs[i][..., -n:])


        spec = torch.cat(decoded_specs, dim=-1)

        spec_min = spec.min().item()
        spec_max = spec.max().item()
        print(f"[DEBUG] spec antes de denormalizar: min={spec_min:.4f}, max={spec_max:.4f}")
        if abs(spec_max - spec_min) < 1e-3:
            print("[WARNING] El espectrograma generado tiene un rango demasiado pequeño")

        try:
            mean_x = self.model.mean_x_buffer.item() if hasattr(self.model.mean_x_buffer, 'item') else float(self.model.mean_x_buffer)
            std_x = self.model.std_x_buffer.item() if hasattr(self.model.std_x_buffer, 'item') else float(self.model.std_x_buffer)
        except Exception as e:
            print(f"[ERROR] Acceso a mean_x o std_x: {e}")
            mean_x, std_x = 0.0, 1.0

        if std_x != 0:
            spec = spec * std_x + mean_x

        spec_db = spec * (db_max - db_min) + db_min
        spec_amp = torch.pow(10.0, spec_db / 20.0)

        if KIND_OF_SPECTROGRAM != 'MEL':
            expected_bins = self.N_FFT // 2 + 1
            if spec_amp.ndim == 2:
                spec_amp = spec_amp.unsqueeze(0)

            if spec_amp.shape[-2] != expected_bins:
                print(f"[WARNING] Redimensionando STFT generado de {spec_amp.shape[-2]} a {expected_bins} bandas para GriffinLim")
                spec_amp = F.interpolate(
                    spec_amp.unsqueeze(1),
                    size=(expected_bins, spec_amp.shape[-1]),
                    mode="bicubic",
                    align_corners=True
                ).squeeze(1)

            for i in range(min(5, expected_bins)):
                spec_amp[0, i, :] *= (i + 1) / 5

            linear_spec = spec_amp.squeeze(0)

            plt.figure(figsize=(12, 4))
            spec_db_vis = torchaudio.transforms.AmplitudeToDB()(linear_spec.unsqueeze(0)).squeeze(0)
            plt.imshow(spec_db_vis.numpy(), aspect='auto', origin='lower', cmap='magma')
            plt.title(f"Espectrograma STFT generado (género {genre_id})")
            plt.xlabel("Frames temporales")
            plt.ylabel("Frecuencia (bins)")
            plt.colorbar(label="Magnitud (dB)")
        else:
            inverse_mel = torchaudio.transforms.InverseMelScale(
                n_stft=self.N_FFT // 2 + 1,
                n_mels=self.NUM_MELS,
                sample_rate=self.SAMPLE_RATE,
            )
            linear_spec = inverse_mel(spec_amp.unsqueeze(0)).squeeze(0)

            plt.figure(figsize=(12, 4))
            plt.imshow(spec_db.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='magma')
            plt.title(f"Espectrograma MEL generado (género {genre_id})")
            plt.xlabel("Frames temporales")
            plt.ylabel("Bandas Mel")
            plt.colorbar(label="Potencia (dB)")

        plt.tight_layout()
        plt.show()

        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            win_length=self.N_FFT,
            power=1.0,
            n_iter=64,
            momentum=0.99,
            length=None,
            rand_init=True
        )

        waveform = griffin_lim(linear_spec)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        torchaudio.save(output_path, waveform, self.SAMPLE_RATE)
        print(f"Audio generado guardado en: {output_path}")

        return waveform, zs