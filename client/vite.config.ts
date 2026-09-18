/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// The API is proxied in development so the browser and the API share an origin. That is
// deliberate: the platform has no CORS middleware, and adding one would widen the server's
// exposure to support a development convenience.
//
// It is proxied under `/api` rather than at the paths the API also serves at the root, because
// this application's own route for a feature is `/features/{id}` -- the same URL. Proxying
// `/features` meant every page load of a feature returned the API's JSON instead of the app:
// deep links, refreshes and new tabs were all broken, while clicking through from the
// dashboard worked because that never leaves the page. The server serves the same routes under
// `/api` for exactly this reason.
const API_PREFIX = '/api';

// The application is served under `/ui`, in development and in production alike, because the
// API owns `/features/{id}` on the same origin and that is also this application's own page
// for a feature. The API's paths are the established contract, so the application moved.
const BASE = '/ui/';

export default defineConfig(({ mode }) => ({
  base: BASE,
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      [API_PREFIX]: {
        target: process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: { outDir: 'dist', sourcemap: mode !== 'production' },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    include: ['tests/**/*.test.{ts,tsx}'],
  },
}));
