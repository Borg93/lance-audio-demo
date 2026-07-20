

export const index = 0;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_layout.svelte.js')).default;
export const universal = {
  "ssr": false,
  "prerender": false
};
export const universal_id = "src/routes/+layout.ts";
export const imports = ["_app/immutable/nodes/0.fmsuO_v0.js","_app/immutable/chunks/CefDVh6V.js","_app/immutable/chunks/BMgjO9vM.js","_app/immutable/chunks/C7JtvbPz.js","_app/immutable/chunks/P2myRLKW.js","_app/immutable/chunks/BLqD86uo.js","_app/immutable/chunks/DRk5mItz.js","_app/immutable/chunks/CKwn_NsA.js","_app/immutable/chunks/C7YjBt91.js","_app/immutable/chunks/CMHk5ZyV.js","_app/immutable/chunks/BcDiIKxj.js","_app/immutable/chunks/CLd2CBe5.js","_app/immutable/chunks/vsDFxLoB.js","_app/immutable/chunks/CX8ud67W.js","_app/immutable/chunks/9wi3y8Nu.js","_app/immutable/chunks/Dzj32VKB.js","_app/immutable/chunks/D7x0IBou.js","_app/immutable/chunks/Db48K74A.js","_app/immutable/chunks/B722GCAU.js"];
export const stylesheets = ["_app/immutable/assets/0.DeBfDYDn.css"];
export const fonts = [];
