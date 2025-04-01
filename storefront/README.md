# EUTERPE 🎼

**Euterpe** es una aplicación para la generación de música personalizada mediante una interfaz web minimalista, conectada a un backend en Flask que integra un modelo GAN para sintetizar audio por género musical.

---

## Requisitos

### Backend (Python)
- Python 3.10+
- [Pipenv](https://pipenv.pypa.io/)
- PyTorch (`torch`, `torchaudio`)

### Frontend (Node.js)
- Node.js 18+
- npm

---

## Instalación y ejecución

### 1. Backend

```bash
cd backend
pipenv install
pipenv run python app.py
```

Esto ejecutará el backend en `http://localhost:5000`.

Asegúrate de que el directorio `static/generated` existe para guardar los audios.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Esto lanza el frontend en `http://localhost:5173`.

---

## Uso

1. Introduce un género musical y pulsa "Generar".
2. Escucha la pista generada por el modelo.
3. Valora la pieza entre 1 y 10.
4. Consulta la sección de ayuda para más detalles.

---
