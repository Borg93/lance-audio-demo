import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    compilerOptions: {
        // Enables `await` directly inside Svelte templates and load functions.
        // Useful for streaming search results progressively.
        experimental: {
            async: true,
        },
    },
    preprocess: vitePreprocess(),
    kit: {
        adapter: adapter({
            // Pure SPA: backend (FastAPI) handles all data; SvelteKit produces
            // static HTML/JS that the existing `frontend/server.ts` Bun proxy
            // can serve unchanged. `200.html` makes unknown routes fall through
            // to client-side routing instead of 404.
            pages: 'build',
            assets: 'build',
            fallback: '200.html',
            precompress: false,
            strict: true,
        }),
        alias: {
            $lib: './src/lib',
            '$lib/*': './src/lib/*',
        },
    },
};

export default config;
