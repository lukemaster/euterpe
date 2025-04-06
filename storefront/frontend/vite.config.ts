import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1", // fuerza IPv4 en dev server
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5000", // evita ::1
        changeOrigin: true,
        secure: false
      }
    }
  }
})
