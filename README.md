# Euterpe

**Euterpe** is a generative system that composes music based on artificial intelligence models. It is part of the Master's Thesis _"Generación de Música Personalizada a través de Modelos Generativos Adversariales"_ by Rafael Luque Tejada (2025).

The system allows training and generation of music using Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN), based on labeled audio datasets.

## Features

- Music generation conditioned by genre.
- Training using VAE or GAN architecture.
- Dataset filtering through a list of valid files.
- Input normalization and reconstruction through inverse STFT.
- Genre-aware decibel scaling.
- Modular design prepared for experimentation.

## Requirements

- Python 3.12.7
- `librosa`, `torch`, `pandas`, `numpy`, `soundfile`, `matplotlib`, and others (see `requirements.txt`)
- CUDA-compatible GPU (optional but recommended)

## Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/yourusername/euterpe.git
cd euterpe
pip install -r requirements.txt
```

## Usage

The entry point is `main.py`. The script expects three arguments:

```bash
python main.py <datasets_path> <valid_files_csv_path> <model_to_train>
```

### Arguments

- `datasets_path`: path to the directory containing the dataset (audio files).
- `valid_files_csv_path`: path to the CSV file listing valid files for training.
- `model_to_train`: `vae` or `gan`, depending on the model to train.

### Example

```bash
python main.py ./data ./valid_files.csv vae
```

## Project Structure

```
euterpe/
├── main.py                # Entry point
├── models/                # Model architectures: VAE, GAN, etc.
├── datasets/              # Dataset utilities and preprocessing
├── training/              # Training loops and evaluation
├── generation/            # Audio generation logic
├── config/                # Environment and parameter configuration
└── utils/                 # Helpers (logging, audio tools...)
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](./LICENSE) file for details.

---

© 2025 Rafael Luque Tejada — lukemaster.master@gmail.com
