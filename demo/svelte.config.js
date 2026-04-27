import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	compilerOptions: {
		experimental: {
			async: true
		}
	},
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			// Separate name so the prerendered `/` page (`index.html`) isn't
			// overwritten. Unknown routes fall through to `200.html` for
			// SPA-style client-side routing.
			fallback: '200.html',
			precompress: false,
			strict: true
		})
	}
};

export default config;
