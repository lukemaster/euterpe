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

class LitVAE(pl.LightningModule):
    BETA_MAX = float(os.environ["BETA_MAX"])
    BETA_WARMUP_EPOCHS = int(os.environ["BETA_WARMUP_EPOCHS"])
    LATENT_DIM = int(os.environ["LATENT_DIM"])
    SAMPLE_RATE = int(os.environ["SAMPLE_RATE"])
    N_FFT = int(os.environ["N_FFT"])
    HOP_LENGTH = int(os.environ["HOP_LENGTH"])
    NUM_MELS = int(os.environ["NUM_MELS"])
    SPEC_TIME_STEPS = int((SAMPLE_RATE * int(os.environ.get('SEGMENT_DURATION'))) / HOP_LENGTH)
    
    class VAE(nn.Module):
        NUM_GENRES = int(os.environ.get('NUM_GENRES'))
        GENRE_EMBEDDING_DIM = (2 ** math.ceil(math.log2(NUM_GENRES)))# * 2 maybe an improvement or not

        class VAE_Encoder(nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.NUM_GENRES = vae.NUM_GENRES
                self.LATENT_DIM = vae.LATENT_DIM
                self.GENRE_EMBEDDING_DIM = vae.GENRE_EMBEDDING_DIM
                self.SPEC_TIME_STEPS = vae.SPEC_TIME_STEPS
                self.genre_embedding = nn.Embedding(self.NUM_GENRES, self.GENRE_EMBEDDING_DIM)

                self.conv1 = nn.Conv2d(1 + self.GENRE_EMBEDDING_DIM, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)

                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)

                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm2d(128)

                self.pool = nn.MaxPool2d(kernel_size=2)

                
                NUM_MELS = int(os.environ.get('NUM_MELS'))
                with torch.no_grad():
                    dummy_spec = torch.zeros(1, 1, NUM_MELS, self.SPEC_TIME_STEPS)
                    dummy_genre = torch.zeros(1, dtype=torch.long)
                    dummy_emb = self.genre_embedding(dummy_genre).view(1, self.GENRE_EMBEDDING_DIM, 1, 1).expand(-1, -1, NUM_MELS, self.SPEC_TIME_STEPS)
                    dummy_input = torch.cat([dummy_spec, dummy_emb], dim=1)

                    dummy = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
                    dummy = self.pool(F.relu(self.bn2(self.conv2(dummy))))
                    dummy = self.pool(F.relu(self.bn3(self.conv3(dummy))))

                    self.flattened_shape = dummy.shape[1:]
                    self.feature_dim = dummy.numel()

                self.fc_mu = nn.Linear(self.feature_dim, self.LATENT_DIM)
                self.fc_logvar = nn.Linear(self.feature_dim, self.LATENT_DIM)

            def forward(self, x, genre):
                x = x[..., :self.SPEC_TIME_STEPS]
                genre_emb = self.genre_embedding(genre).view(genre.size(0), self.GENRE_EMBEDDING_DIM, 1, 1)
                genre_emb = genre_emb.expand(-1, -1, x.size(2), x.size(3))
                x = torch.cat([x, genre_emb], dim=1)

                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                x = self.pool(F.relu(self.bn2(self.conv2(x))))
                x = self.pool(F.relu(self.bn3(self.conv3(x))))
                x = x.view(x.size(0), -1)

                mu = self.fc_mu(x)
                logvar = self.fc_logvar(x)
                logvar = torch.clamp(logvar, min=-10.0, max=10.0)
                
                return mu, logvar

        class VAE_Decoder(nn.Module):
            def __init__(self, vae, feature_dim, flattened_shape):
                super().__init__()
                self.NUM_GENRES = vae.NUM_GENRES
                self.GENRE_EMBEDDING_DIM = vae.GENRE_EMBEDDING_DIM
                self.LATENT_DIM = vae.LATENT_DIM
                self.feature_dim = feature_dim
                self.flattened_shape = flattened_shape

                self.genre_embedding = nn.Embedding(self.NUM_GENRES, self.GENRE_EMBEDDING_DIM)
                self.fc = nn.Linear(self.LATENT_DIM + self.GENRE_EMBEDDING_DIM, feature_dim)

                self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

                self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
                self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
                self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

                self.output_pad = nn.ConstantPad2d((0, 5), 0)

            def forward(self, z, genre):
                genre_emb = self.genre_embedding(genre)
                z = torch.cat([z, genre_emb], dim=1)

                x = self.fc(z)
                x = x.view(z.size(0), *self.flattened_shape)


                x = x.permute(0, 2, 1, 3)
                B, H, C, W = x.shape
                x = x.reshape(B * H, C, W).permute(0, 2, 1)
                x, _ = self.lstm(x)
                x = x.permute(0, 2, 1).reshape(B, H, C, W)
                x = x.permute(0, 2, 1, 3)

                x = F.relu(self.deconv1(x))
                x = F.relu(self.deconv2(x))
                
                x = self.deconv3(x)
                x = self.output_pad(x)

                
                # if not self.training:
                #     import matplotlib.pyplot as plt

                #     flat = x.detach().cpu().flatten()
                #     x_min = flat.min().item()
                #     x_max = flat.max().item()
                #     x_mean = flat.mean().item()
                #     x_std = flat.std().item()
                #     sat_low = (flat <= -0.95).float().mean().item()
                #     sat_high = (flat >= 0.95).float().mean().item()

                #     print(f"[DEBUG] Decoder output stats -> min: {x_min:.4f}, max: {x_max:.4f}, mean: {x_mean:.4f}, std: {x_std:.4f}")
                #     print(f"[DEBUG] Saturación -> <= -0.95: {sat_low:.2%}, >= 0.95: {sat_high:.2%}")

                #     plt.figure(figsize=(6, 4))
                #     plt.hist(flat.numpy(), bins=50, color='purple', alpha=0.75)
                #     plt.title("Histograma de valores del decoder")
                #     plt.xlabel("Valor")
                #     plt.ylabel("Frecuencia")
                #     plt.grid(True)
                #     plt.tight_layout()
                #     plt.show()

                return x
                
        def __init__(self, litVAE):
            super().__init__()
            self.SPEC_TIME_STEPS = litVAE.SPEC_TIME_STEPS
            self.LATENT_DIM = litVAE.LATENT_DIM
            self.encoder = self.VAE_Encoder(self)
            self.decoder = self.VAE_Decoder(self,self.encoder.feature_dim, self.encoder.flattened_shape)
            self.apply(self.init_weights)
          
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
            # print(f"[ENCODER IN] x mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
            mu, logvar = self.encoder(x, genre)
            # print(f"[LATENT] mu mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")
            # print(f"[LATENT] logvar mean: {logvar.mean().item():.4f}, std: {logvar.std().item():.4f}")

            z = self.reparameterize(mu, logvar)
            x_hat = self.decoder(z, genre)
            if not torch.isfinite(x_hat).all():
                print("[ERROR] x_hat contiene NaNs")

            # print(f"[DECODER OUT] x_recon mean: {x_hat.mean().item():.4f}, std: {x_hat.std().item():.4f}")


            if not torch.isfinite(x).all():
                print("[ERROR] Input contiene NaNs o infinitos")
            if not torch.isfinite(mu).all():
                print("[ERROR] mu contiene NaNs")
            if not torch.isfinite(logvar).all():
                print("[ERROR] logvar contiene NaNs")

            return x_hat, mu, logvar

    def __init__(self):
        super().__init__()
        self.model = self.VAE(self)
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