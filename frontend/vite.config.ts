import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': 'http://localhost:80',
      '/pipeline': 'http://localhost:80',
    }
  },
  build: { outDir: 'dist' }
})


