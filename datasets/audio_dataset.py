import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dotenv import load_dotenv
load_dotenv('./VIU/09MIAR/src/vae/.env')

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE'))
        self.N_FFT = int(os.environ.get('N_FFT'))
        self.HOP_LENGTH = int(os.environ.get('HOP_LENGTH'))
        self.NUM_MELS = int(os.environ.get('NUM_MELS'))

        self.SEGMENT_DURATION = int(os.environ.get('SEGMENT_DURATION'))
        self.segment_samples = self.SEGMENT_DURATION * self.SAMPLE_RATE
        adjustment = 0
        if (self.segment_samples + (self.N_FFT - self.HOP_LENGTH)) % self.HOP_LENGTH != 0:
            adjustment = 1
        self.input_w = (self.segment_samples + (self.N_FFT - self.HOP_LENGTH)) // self.HOP_LENGTH + adjustment

        self.file_paths = list(file_paths)
        self.labels = list(labels)
        assert len(self.file_paths) == len(self.labels), "Labels amount don't match with files amount"

        self.SPEC_TIME_STEPS = int(os.environ.get('SPEC_TIME_STEPS'))
        self.segments_per_track = int(int(os.environ.get('TOTAL_DURATION')) / int(os.environ.get('SEGMENT_DURATION')))

        self.kind_of_spectrogram = os.environ.get('KIND_OF_SPECTROGRAM')
        self.num_freq_bins = None

        if self.kind_of_spectrogram != "MEL":
            def stft(waveform):
                spec = torchaudio.transforms.Spectrogram(
                    n_fft=self.N_FFT,
                    hop_length=self.HOP_LENGTH,
                    power=2.0
                )(waveform)
                self.num_freq_bins = self.N_FFT // 2 + 1
                return torchaudio.transforms.AmplitudeToDB()(spec)
            self.spec_transform = stft
            
        else:
            def mel(waveform):
                spec = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.SAMPLE_RATE,
                    n_fft=self.N_FFT,
                    hop_length=self.HOP_LENGTH,
                    n_mels=self.NUM_MELS
                )(waveform)
                self.num_freq_bins = self.NUM_MELS
                return torchaudio.transforms.AmplitudeToDB()(spec)
            self.spec_transform = mel
            


    def __len__(self):
        return len(self.file_paths) * self.segments_per_track
    
    def jump_next_file(self, index):
        # print('Rolling to next file...')
        del self.file_paths[index]
        return self.__getitem__((index + 1) % len(self.file_paths))
            

    def __getitem__(self, index):
        try:
            segment_samples = self.segment_samples

            file_index = index // self.segments_per_track
            segment_index = index % self.segments_per_track

            filepath = self.file_paths[file_index]
            genre_idx = self.labels[file_index]

            waveform, sr = torchaudio.load(filepath)

            if sr > self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.SAMPLE_RATE)
                waveform = resampler(waveform)
                sr = self.SAMPLE_RATE
            elif sr < self.SAMPLE_RATE:
                raise ValueError(f"{filepath} descartado por sample rate bajo: {sr}")

            waveform = waveform.mean(dim=0, keepdim=True)
            pad_len = (self.N_FFT - self.HOP_LENGTH) // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode='reflect')

            start = segment_index * segment_samples
            end = start + segment_samples
            waveform_segment = waveform[:, start:end]
            waveform_segment = F.pad(waveform_segment, (pad_len, pad_len), mode='reflect')

            if waveform_segment.size(1) < segment_samples:
                waveform_segment = F.pad(waveform_segment, (0, segment_samples - waveform_segment.size(1)))

            # if self.kind_of_spectrogram == "STFT":
            #     spec_transform = nn.Sequential(
            #         torchaudio.transforms.Spectrogram(
            #             n_fft=self.N_FFT,
            #             hop_length=self.HOP_LENGTH,
            #             power=2.0
            #         ),
            #         torchaudio.transforms.AmplitudeToDB()
            #     )
            #     num_freq_bins = self.N_FFT // 2 + 1
            # else:
            #     spec_transform = nn.Sequential(
            #         torchaudio.transforms.MelSpectrogram(
            #             sample_rate=self.SAMPLE_RATE,
            #             n_fft=self.N_FFT,
            #             hop_length=self.HOP_LENGTH,
            #             n_mels=self.NUM_MELS
            #         ),
            #         torchaudio.transforms.AmplitudeToDB()
            #     )
            #     num_freq_bins = self.NUM_MELS

            self.SPEC_TIME_STEPS = int((segment_samples + 2 * pad_len) / self.HOP_LENGTH) + 1

            segment_spec = self.spec_transform(waveform_segment)
            # segment_spec = spec_transform(waveform_segment)
            spec_mean = segment_spec.mean()
            spec_std = segment_spec.std()

            if spec_std < 1e-5:
                segment_spec = torch.zeros_like(segment_spec)
            else:
                segment_spec = (segment_spec - spec_mean) / spec_std

            # Recorte/padding a tamaño fijo en eje temporal
            if segment_spec.shape[-1] < self.SPEC_TIME_STEPS:
                pad_amount = self.SPEC_TIME_STEPS - segment_spec.shape[-1]
                segment_spec = F.pad(segment_spec, (0, pad_amount))
            else:
                segment_spec = segment_spec[..., :self.SPEC_TIME_STEPS]

            # Adaptar la dimensión de frecuencias si se desea forzar compatibilidad con red entrenada con MEL
            if self.kind_of_spectrogram == "STFT" and self.num_freq_bins != self.NUM_MELS:
                segment_spec = F.interpolate(segment_spec.unsqueeze(0), size=(self.NUM_MELS, self.SPEC_TIME_STEPS), mode="bilinear", align_corners=False).squeeze(0)

            return segment_spec, genre_idx.clone().detach().to(torch.long)

        except Exception as e:
            print(f"caught exception: {e}")
            return self.jump_next_file(index)
        except RuntimeError as e:
            print(f"[ERROR] Archivo dañado o ilegible: {filepath} - {e}")
            return self.jump_next_file(index)

    def __gatitem__(self, index):
        try:
            segment_samples = self.segment_samples

            file_index = index // self.segments_per_track
            segment_index = index % self.segments_per_track

            filepath = self.file_paths[file_index]
            genre_idx = self.labels[file_index]

            waveform, sr = torchaudio.load(filepath)

            if sr > self.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.SAMPLE_RATE)
                waveform = resampler(waveform)
                sr = self.SAMPLE_RATE
            elif sr < self.SAMPLE_RATE:
                raise ValueError(f"{filepath} descartado por sample rate bajo: {sr}")

            waveform = waveform.mean(dim=0, keepdim=True)

            pad_len = (self.N_FFT - self.HOP_LENGTH) // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode='reflect')

            start = segment_index * segment_samples
            end = start + segment_samples
            waveform_segment = waveform[:, start:end]
            waveform_segment = F.pad(waveform_segment, (pad_len, pad_len), mode='reflect')

            if waveform_segment.size(1) < segment_samples:
                waveform_segment = F.pad(waveform_segment, (0, segment_samples - waveform_segment.size(1)))

            # MelSpectrogram completo
            # print(f"[INFO] Track index {index}: full_spec.shape before normalization = {waveform.shape}")
            
            # full_spec = torchaudio.transforms.MelSpectrogram(
            #     sample_rate=self.SAMPLE_RATE,
            #     n_fft=self.N_FFT,
            #     hop_length=self.HOP_LENGTH,
            #     n_mels=self.NUM_MELS
            # )(waveform)
            # full_spec = torchaudio.transforms.AmplitudeToDB()(full_spec)

            full_spec = self.spec_transform(waveform)

            fs_mean = full_spec.mean()
            fs_std = full_spec.std()

            if fs_std < 1e-5:
                full_spec = torch.zeros_like(full_spec)
            else:
                full_spec = (full_spec - fs_mean) / fs_std


            # MelSpectrogram del segmento
            # spec = torchaudio.transforms.MelSpectrogram(
            #     sample_rate=self.SAMPLE_RATE,
            #     n_fft=self.N_FFT,
            #     hop_length=self.HOP_LENGTH,
            #     n_mels=self.NUM_MELS
            # )(waveform_segment)
            # spec = torchaudio.transforms.AmplitudeToDB()(spec)

            spec = self.spec_transform(waveform_segment)



            # if spec.shape[-1] != 2510:
            #     self.jump_next_file(index)
            # print(f"[INFO] Segmento index {index}: spec.shape = {spec.shape}")




            spec_mean = spec.mean()
            spec_std = spec.std()

            if spec_std < 1e-5:
                spec = torch.zeros_like(spec)
            else:
                spec = (spec - spec_mean) / spec_std




            # segment_start_sec = start / self.SAMPLE_RATE
            # segment_end_sec = end / self.SAMPLE_RATE

            # frame_start = start // self.HOP_LENGTH
            # frame_end = frame_start + spec.shape[-1]

            # fig, axs = plt.subplots(2, 1, figsize=(12, 8))

            # total_duration_sec = waveform.size(1) / self.SAMPLE_RATE
            # axs[0].imshow(full_spec.squeeze().cpu(), aspect='auto', origin='lower', cmap='magma', extent=[0, total_duration_sec * 1000, 0, self.NUM_MELS])
            # axs[0].set_title(f"Espectrograma Mel completo (Duración: {total_duration_sec * 1000:.0f} ms)")
            # axs[0].set_ylabel("Bandas Mel")
            # axs[0].add_patch(patches.Rectangle((segment_start_sec * 1000, 0), (segment_end_sec - segment_start_sec) * 1000, self.NUM_MELS,
            #                                    linewidth=2, edgecolor='red', facecolor='none'))

            # axs[1].imshow(spec.squeeze().cpu(), aspect='auto', origin='lower', cmap='magma', extent=[segment_start_sec * 1000, segment_end_sec * 1000, 0, self.NUM_MELS])
            # axs[1].set_title(f"Segmento extraído: {segment_start_sec:.2f}s a {segment_end_sec:.2f}s")
            # axs[1].set_xlabel(f"Tiempo: {int(segment_start_sec * 1000)} ms - {int(segment_end_sec * 1000)} ms")
            # axs[1].set_ylabel("Bandas Mel")

            # plt.tight_layout()
            # plt.show()

            # print(f"[GETITEM] full_spec mean: {full_spec.mean().item():.4f}, std: {full_spec.std().item():.4f}")
            # print(f"[GETITEM] spec mean: {spec.mean().item():.4f}, std: {spec.std().item():.4f}")



            return spec, genre_idx.clone().detach().to(torch.long)
            return spec, torch.tensor(genre_idx, dtype=torch.long)

        except Exception as e:
            print(f'''caught exception: {e}''')
            return self.jump_next_file(index)
        except RuntimeError as e:
            print(f"[ERROR] Archivo dañado o ilegible: {filepath} - {e}")
            return self.jump_next_file(index)
