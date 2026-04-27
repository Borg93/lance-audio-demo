// Fully static SPA: no SSR, all routes pre-rendered to HTML shells at build
// time, client hydrates. Works on any static host (Hugging Face Spaces,
// GitHub Pages, Netlify, S3, etc.).
export const ssr = false;
export const prerender = true;
