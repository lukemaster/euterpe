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


from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from datetime import datetime
from typing import Dict, TypedDict
import os
import csv
from generator.generator import MusicGenerator  # GAN real o dummy

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

generator = MusicGenerator()
generator.eval()

CSV_PATH = 'register.csv'

class PayloadGenerator(TypedDict):
    genre: str

class ResponseGenerator(TypedDict):
    id: int
    name: str
    genre: str
    url: str

class HelpResponse(TypedDict):
    texto: str

def generation_register(genre: str, noise: float, file_name: str) -> None:
    existe = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(['timestamp', 'genre', 'noise', 'archivo'])
        writer.writerow([datetime.utcnow().isoformat(), genre, noise, file_name])

@app.route('/api/generate', methods=['POST'])
def generar() -> Response:
    data: PayloadGenerator = request.get_json()
    genre: str = data.get('genre', 'unknown')
    noise: float = round(os.urandom(2)[0] / 255, 4)
    timestamp: int = int(datetime.utcnow().timestamp())
    track_name: str = f'{genre}_{timestamp}.mp3'
    ruta_absoluta: str = os.path.join('static/generated', track_name)

    try:
        generator.generate(genre, noise, ruta_absoluta)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    generation_register(genre, noise, track_name)

    response: ResponseGenerator = {
        'id': timestamp,
        'name': track_name,
        'genre': genre,
        'url': f'/static/generated/{track_name}'
    }
    return jsonify(response)

@app.route('/api/rate', methods=['POST'])
def rate() -> tuple[str, int]:
    # No se almacena la valoración en CSV en esta versión TODO
    return ('', 204)

@app.route('/api/help')
def ayuda() -> Response:
    texto = '''
    Bienvenido a Euterpe. Selecciona un género musical, pulsa Generar, escucha y valora.
    Cada pieza es única gracias al noise aleatorio con el que se crea.
    Tus valoraciones ayudan a mejorar el sistema.
    '''
    return jsonify({"texto": texto.strip()})


@app.route('/api/genres')
def genres() -> Response:
    return jsonify({'genres':['hip-hop','jazz','rock','pop','blues']})

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
