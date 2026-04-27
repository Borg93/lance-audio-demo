import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
    plugins: [tailwindcss(), sveltekit()],
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
