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
import io
import csv
import uuid
import base64
from datetime import datetime
from typing import Dict, TypedDict

import torch

from flask_cors import CORS
from flask import Flask, request, jsonify, Response, send_file, after_this_request

from generator.generator import MusicGenerator  # GAN real o dummy

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

generator = MusicGenerator()
generator.eval()

csv.field_size_limit(10**8)  # aumenta a ~100 MB por campo
CSV_PATH = 'register.csv'

genres_arr = ['hip-hop','jazz','rock','pop','blues']

class PayloadGenerator(TypedDict):
    genre: str

class ResponseGenerator(TypedDict):
    id: int
    name: str
    genre: str
    url: str

class HelpResponse(TypedDict):
    texto: str

def generation_register(uuid: str, genre: str, noise: float, file_name: str) -> None:
    existe = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(['id', 'genre', 'noise', 'archivo'])
        writer.writerow([uuid, genre, noise, file_name])

@app.route('/api/generate', methods=['POST'])
def generar() -> Response:
    data: PayloadGenerator = request.get_json()
    genre: str = data.get('genre', 'unknown')
    timestamp: int = int(datetime.utcnow().timestamp())
    track_name: str = f'{genre}_{timestamp}.mp3'

    try:
        genre_idx = genres_arr.index(genre)
        noise, mp3_io = generator.generate(genre_idx,'static/generated',track_name)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    buffer = io.BytesIO()
    torch.save(noise, buffer)
    buffer.seek(0)
    csv_string = base64.b64encode(noise.cpu().numpy().astype("float32").tobytes()).decode("utf-8")
    
    track_id = str(uuid.uuid4())
    generation_register(track_id, genre, csv_string, track_name)

    # @after_this_request TODO
    # def remove_file(response):
    #     try:
    #         os.remove(file_path)
    #     except Exception as e:
    #         print(f"{file_path}: {e}")
    #     return response


    response = send_file(
        mp3_io,
        mimetype="audio/mpeg",
        as_attachment=True,
        download_name=track_name
    )
    response.headers["X-Track-ID"] = uuid
    return response

@app.route('/api/rate', methods=['POST'])
def rate() -> tuple[str, int]:
    data = request.get_json()
    track_id = data.get("track_id")
    rating = data.get("rating")

    if not track_id or rating is None:
        return ("Missing 'id' or 'rating'", 400)

    updated = False
    rows = []

    with open(CSV_PATH, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames or []
        if "rating" not in fieldnames:
            fieldnames.append("rating")

        for row in reader:
            if row["id"] == track_id:
                row["rating"] = str(rating)
                updated = True
            rows.append(row)

    if not updated:
        return ("ID not found", 404)

    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return ('', 204)

@app.route('/api/help')
def ayuda() -> Response:
    texto = '''
    Bienvenido a Euterpe. Selecciona un género musical, pulsa Generar, escucha y valora.
    Cada pieza es única gracias al ruido aleatorio con el que se crea.
    Tus valoraciones ayudan a mejorar el sistema.
    '''
    return jsonify({"texto": texto.strip()})


@app.route('/api/genres')
def genres() -> Response:
    return jsonify({'genres':genres_arr})

@app.route('/api/history')
def historial() -> Response:
    if not os.path.exists(CSV_PATH):
        return jsonify([])
    with open(CSV_PATH, newline='') as f:
        reader = list(csv.DictReader(f))
        ultimos = reader[-10:] # last 10 lines
    return jsonify(ultimos)

if __name__ == '__main__':
    os.makedirs('static/generated', exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
