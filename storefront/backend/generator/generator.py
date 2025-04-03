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
