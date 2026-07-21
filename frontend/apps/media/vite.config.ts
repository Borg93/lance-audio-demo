import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  // OpenCV.js is dynamically imported by the scissors/magnetic tools (8MB wasm,
  // lazy-loaded on first activation). Prebundle it at server start — otherwise the
  // first in-session import triggers a dep re-optimize + full page reload mid-annotation.
  optimizeDeps: {
    include: ['@techstark/opencv-js'],
  },
  server: {
    // During `bun run dev`, proxy /api/* straight to the FastAPI backend
    // so the SvelteKit dev server can be tested without launching the
    // Bun proxy. Production build still goes through frontend/server.ts.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
