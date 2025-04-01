from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from datetime import datetime
from typing import Dict, TypedDict
import os
import csv
from generator.generator import MusicGenerator  # GAN real o dummy

app = Flask(__name__)
CORS(app)

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

@app.route('/api/generar', methods=['POST'])
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

@app.route('/api/valorar', methods=['POST'])
def valorar() -> tuple[str, int]:
    # No se almacena la valoración en CSV en esta versión TODO
    return ('', 204)

@app.route('/api/help')
def ayuda() -> Response:
    texto = '''
    Bienvenido a Euterpe. Selecciona un género musical, pulsa Generar, escucha y valora.
    Cada pieza es única gracias al noise aleatorio con el que se crea.
    Tus valoraciones ayudan a mejorar el sistema.
    '''
    response: HelpResponse = {'texto': texto.strip()}
    return jsonify(response)

@app.route('/api/genres')
def genres() -> Response:
    return jsonify(['jazz', 'rock', 'clásica', 'electrónica', 'lo-fi', 'ambient'])

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
    app.run(debug=True)
