

export const index = 0;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_layout.svelte.js')).default;
export const universal = {
  "ssr": false,
  "prerender": false
};
export const universal_id = "src/routes/+layout.ts";
export const imports = ["_app/immutable/nodes/0.Cw9wrwys.js","_app/immutable/chunks/-PMr3y2J.js","_app/immutable/chunks/dsdhJolP.js","_app/immutable/chunks/CrCJhyX1.js","_app/immutable/chunks/i-hiIcg-.js","_app/immutable/chunks/hzFSh3HY.js","_app/immutable/chunks/8crZtJf2.js","_app/immutable/chunks/CvQ65UvP.js","_app/immutable/chunks/ZzQKDqew.js","_app/immutable/chunks/B2xZ8vQm.js","_app/immutable/chunks/CCaiLPw2.js","_app/immutable/chunks/BTZYp3ed.js","_app/immutable/chunks/DjeedS84.js","_app/immutable/chunks/BdZvPen0.js","_app/immutable/chunks/hhV791lv.js","_app/immutable/chunks/Bvi-xls8.js","_app/immutable/chunks/SZ7DvS8J.js","_app/immutable/chunks/DKcmKMNe.js","_app/immutable/chunks/jerKUr4u.js","_app/immutable/chunks/BHLHc1JI.js","_app/immutable/chunks/DxUrc7iO.js"];
export const stylesheets = ["_app/immutable/assets/0.CTIWbJoV.css"];
export const fonts = [];
