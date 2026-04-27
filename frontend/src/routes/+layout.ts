// Pure SPA: render in the browser only. SvelteKit's adapter-static still
// outputs HTML/JS for the client, but no SSR step runs at build time.
export const prerender = true;
export const ssr = false;
