import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiProxyTarget = (
    process.env.VITE_DEV_API_PROXY ||
    env.VITE_DEV_API_PROXY ||
    'http://localhost:8000'
  ).replace(/\/$/, '')
  const pipelineProxyTarget = (
    process.env.VITE_DEV_PIPELINE_PROXY ||
    env.VITE_DEV_PIPELINE_PROXY ||
    'http://localhost:8001'
  ).replace(/\/$/, '')

  return {
    build: {
      outDir: 'dist',
      assetsDir: 'assets',
    },
    server: {
      host: '0.0.0.0',
      port: 5173,
      proxy: {
        '/api': {
          target: apiProxyTarget,
          changeOrigin: true,
        },
        '/pipeline': {
          target: pipelineProxyTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
