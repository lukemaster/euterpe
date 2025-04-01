import os
import time
import shutil
from typing import Literal

OUT_PATH = "static/generated"

class MusicGenerator:
    def __init__(self) -> None:
        self.generos = ["jazz", "rock", "clásica", "electrónica", "lo-fi", "ambient"]

    def eval(self) -> None:
        pass  # En un modelo real esto activaría el modo evaluación

    def generate(self, genero: str, ruido: float, destino: str) -> None:
        if genero not in self.generos:
            raise ValueError(f"Género no válido: {genero}")
        
        # time generation simulation
        time.sleep(1.5)

        # ensure path exists
        os.makedirs(os.path.dirname(destino), exist_ok=True)

        # take a fake mp3 like base
        plantilla = "dummy_audio.mp3"
        origen = os.path.join(os.path.dirname(__file__), plantilla)
        shutil.copy(origen, destino)
