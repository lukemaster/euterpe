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
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from typing import Optional, Dict, TypedDict, List
import os
from generator.generator import MusicGenerator  # GAN real o dummy

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///euterpe.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
generator = MusicGenerator()
generator.eval()

class Solicitud(db.Model):
    id: int = db.Column(db.Integer, primary_key=True)
    timestamp: datetime = db.Column(db.DateTime, default=datetime.utcnow)
    genero: str = db.Column(db.String(100))
    ruido: float = db.Column(db.Float)
    puntuacion: Optional[int] = db.Column(db.Integer)
    ruta_pieza: str = db.Column(db.String(200))

class GenerarPayload(TypedDict):
    genero: str

class GenerarResponse(TypedDict):
    id: int
    nombre: str
    genero: str
    url: str

class AyudaResponse(TypedDict):
    texto: str

@app.route('/api/generar', methods=['POST'])
def generar() -> Response:
    data: GenerarPayload = request.get_json()
    genero: str = data.get('genero', 'desconocido')
    ruido: float = round(os.urandom(2)[0] / 255, 4)
    timestamp: int = int(datetime.utcnow().timestamp())
    nombre_pista: str = f'{genero}_{timestamp}.mp3'
    ruta_absoluta: str = os.path.join('static/generated', nombre_pista)

    try:
        generator.generate(genero, ruido, ruta_absoluta)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    nueva = Solicitud(genero=genero, ruido=ruido, ruta_pieza=f'/static/generated/{nombre_pista}')
    db.session.add(nueva)
    db.session.commit()

    response: GenerarResponse = {
        'id': nueva.id,
        'nombre': nombre_pista,
        'genero': genero,
        'url': f'/static/generated/{nombre_pista}'
    }
    return jsonify(response)

@app.route('/api/valorar', methods=['POST'])
def valorar() -> tuple[str, int]:
    data: Dict[str, int] = request.get_json()
    solicitud: Optional[Solicitud] = Solicitud.query.get(data.get('id'))
    if solicitud:
        solicitud.puntuacion = data.get('valoracion')
        db.session.commit()
    return ('', 204)

@app.route('/api/ayuda')
def ayuda() -> Response:
    texto = '''
    Bienvenido a Euterpe. Selecciona un género musical, pulsa Generar, escucha y valora.
    Cada pieza es única gracias al ruido aleatorio con el que se crea.
    Tus valoraciones ayudan a mejorar el sistema.
    '''
    response: AyudaResponse = {'texto': texto.strip()}
    return jsonify(response)

@app.route('/api/generos')
def generos() -> Response:
    return jsonify(['jazz', 'rock', 'clásica', 'electrónica', 'lo-fi', 'ambient'])

@app.route('/api/historial')
def historial() -> Response:
    ultimas: List[Solicitud] = Solicitud.query.order_by(Solicitud.timestamp.desc()).limit(10).all()
    response = [
        {
            'id': s.id,
            'genero': s.genero,
            'ruido': s.ruido,
            'puntuacion': s.puntuacion,
            'fecha': s.timestamp.isoformat(),
            'ruta': s.ruta_pieza
        }
        for s in ultimas
    ]
    return jsonify(response)

if __name__ == '__main__':
    os.makedirs('static/generated', exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
