/**
 * Dummy single-file labeler. Not part of the real app — lives in tools/.
 *
 *   cd tools/labeler
 *   bun run labeler.ts --root ../../input --port 3999
 *   # open http://localhost:3999
 *
 * What it does:
 *   - lists every input/<lang>/ subdir with file count
 *   - click one → iterate files one at a time with video player
 *   - keyboard: 1=sv  2=en  3=skip  o=other  n=next  p=prev
 *   - "action" moves the file to input/<target>/ and advances
 */

import { existsSync, readdirSync, statSync, renameSync, mkdirSync } from "node:fs";
import { resolve, join, basename, normalize } from "node:path";

const args = Object.fromEntries(
    process.argv.slice(2).map((a, i, all) =>
        a.startsWith("--") ? [a.slice(2), all[i + 1]] : null,
    ).filter((x): x is [string, string] => x !== null),
);

const ROOT = resolve(args.root ?? "../../input");
const PORT = Number(args.port ?? 3999);

if (!existsSync(ROOT)) {
    console.error(`root does not exist: ${ROOT}`);
    process.exit(1);
}

/** Confine paths to ROOT so a bad request can't escape the tree. */
function safeJoin(...parts: string[]): string | null {
    const abs = normalize(resolve(ROOT, ...parts));
    return abs.startsWith(ROOT) ? abs : null;
}

function listDirs() {
    return readdirSync(ROOT)
        .filter((n) => statSync(join(ROOT, n)).isDirectory())
        .map((n) => ({
            lang: n,
            count: readdirSync(join(ROOT, n)).filter((f) => f.endsWith(".mp4")).length,
        }))
        .sort((a, b) => a.lang.localeCompare(b.lang));
}

function listFiles(lang: string) {
    const d = safeJoin(lang);
    if (!d || !existsSync(d) || !statSync(d).isDirectory()) return null;
    return readdirSync(d).filter((f) => f.endsWith(".mp4")).sort();
}

function serveVideo(req: Request, path: string): Response {
    const size = statSync(path).size;
    const range = req.headers.get("range");
    const file = Bun.file(path);
    if (range) {
        const m = range.match(/^bytes=(\d*)-(\d*)$/);
        if (!m) return new Response("bad range", { status: 416 });
        const start = m[1] ? Number(m[1]) : 0;
        const end = m[2] ? Number(m[2]) : size - 1;
        if (start > end || end >= size) {
            return new Response("range oob", { status: 416, headers: { "Content-Range": `bytes */${size}` } });
        }
        return new Response(file.slice(start, end + 1), {
            status: 206,
            headers: {
                "Content-Type": "video/mp4",
                "Content-Length": String(end - start + 1),
                "Content-Range": `bytes ${start}-${end}/${size}`,
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-store",
            },
        });
    }
    return new Response(file, {
        headers: {
            "Content-Type": "video/mp4",
            "Content-Length": String(size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        },
    });
}

Bun.serve({
    port: PORT,
    async fetch(req) {
        const url = new URL(req.url);

        if (url.pathname === "/") {
            return new Response(HTML, { headers: { "Content-Type": "text/html; charset=utf-8" } });
        }

        if (url.pathname === "/api/dirs") {
            return Response.json(listDirs());
        }

        let m = url.pathname.match(/^\/api\/dir\/([A-Za-z0-9_-]+)$/);
        if (m) {
            const files = listFiles(m[1]!);
            if (!files) return new Response("not found", { status: 404 });
            return Response.json(files);
        }

        m = url.pathname.match(/^\/video\/([A-Za-z0-9_-]+)\/(.+\.mp4)$/);
        if (m) {
            const p = safeJoin(m[1]!, basename(m[2]!));
            if (!p || !existsSync(p)) return new Response("not found", { status: 404 });
            return serveVideo(req, p);
        }

        if (url.pathname === "/api/move" && req.method === "POST") {
            const body = await req.json() as { from_lang: string; to_lang: string; filename: string };
            const src = safeJoin(body.from_lang, basename(body.filename));
            const dstDir = safeJoin(body.to_lang);
            if (!src || !dstDir || !existsSync(src)) {
                return new Response("src not found", { status: 404 });
            }
            mkdirSync(dstDir, { recursive: true });
            const dst = join(dstDir, basename(body.filename));
            renameSync(src, dst);
            return Response.json({ moved: dst.replace(ROOT + "/", "") });
        }

        return new Response("not found", { status: 404 });
    },
});

console.log(`labeler ready: http://localhost:${PORT}  (root=${ROOT})`);


// ───────────── everything below is the single-file UI ─────────────────
const HTML = `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" />
<title>labeler</title>
<style>
:root { color-scheme: dark; --bg:#0e0f13; --panel:#15171d; --border:#272a33; --muted:#9aa0a8; --text:#e8e9ee; --accent:#6ca6ff; --ok:#7fd07f; --warn:#e7bd54; }
*{box-sizing:border-box}
body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
header{padding:12px 20px;border-bottom:1px solid var(--border);display:flex;gap:16px;align-items:center;background:var(--panel)}
header h1{margin:0;font-size:15px;font-weight:600}
header a{color:var(--accent);text-decoration:none;margin-left:12px;font-size:13px}
.path{color:var(--muted);font-size:13px;font-family:ui-monospace,Menlo,monospace;margin-left:auto}
main{padding:16px 20px;max-width:1100px;margin:0 auto}
.dirs{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px}
.tile{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px 12px;cursor:pointer;text-align:center;transition:transform .1s,border-color .12s;color:inherit;text-decoration:none;display:block}
.tile:hover{border-color:var(--accent);transform:translateY(-1px)}
.tile .lang{font-size:18px;font-weight:700;font-family:ui-monospace,Menlo,monospace}
.tile .count{font-size:12px;color:var(--muted);margin-top:4px}
.bar{display:flex;gap:16px;align-items:center;justify-content:space-between;padding:8px 0 14px}
.bar .title{font-size:14px;font-family:ui-monospace,Menlo,monospace}
.bar .prog{font-size:13px;color:var(--muted)}
video{width:100%;max-height:60vh;background:#000;border-radius:8px}
.ctl{display:flex;gap:10px;padding:14px 0;flex-wrap:wrap}
.btn{flex:1;min-width:100px;padding:12px 14px;font-size:15px;font-weight:600;border:1px solid var(--border);background:var(--panel);color:var(--text);border-radius:8px;cursor:pointer}
.btn:hover{border-color:var(--accent)}
.btn.ok{border-color:var(--ok);color:var(--ok)}
.btn.en{border-color:var(--accent);color:var(--accent)}
.btn.mix{border-color:#c48fff;color:#c48fff}
.btn.skip{border-color:var(--warn);color:var(--warn)}
.btn .kbd{font-size:11px;color:var(--muted);margin-left:6px;font-family:ui-monospace,Menlo,monospace}
.empty{padding:40px;text-align:center;color:var(--muted)}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:var(--panel);border:1px solid var(--border);color:var(--text);padding:8px 14px;border-radius:6px;font-size:13px;opacity:0;transition:opacity .2s}
.toast.show{opacity:1}
</style></head><body>
<header>
  <h1>labeler <span style="color:var(--muted);font-weight:400">— click through + relabel</span></h1>
  <a href="/">home</a>
  <span class="path" id="path"></span>
</header>
<main>
  <section id="dirs-view"></section>
  <section id="lab-view" hidden>
    <div class="bar"><div class="title" id="title">—</div><div class="prog" id="prog">—</div></div>
    <video id="video" controls autoplay muted playsinline></video>
    <div class="ctl">
      <button class="btn ok"  data-a="sv">🇸🇪 Swedish <span class="kbd">1</span></button>
      <button class="btn en"  data-a="en">🇬🇧 English <span class="kbd">2</span></button>
      <button class="btn mix" data-a="mixed">🌐 Mixed / non-en <span class="kbd">4</span></button>
      <button class="btn skip" data-a="skip">skip <span class="kbd">3</span></button>
      <button class="btn"      data-a="other">other… <span class="kbd">o</span></button>
      <button class="btn"      data-a="prev">◀ prev <span class="kbd">p</span></button>
      <button class="btn"      data-a="next">next ▶ <span class="kbd">n</span></button>
    </div>
  </section>
</main>
<div class="toast" id="toast"></div>
<script>
const qp = new URLSearchParams(location.search);
const dir = qp.get("dir");
const pathEl = document.getElementById("path");
const dirsV = document.getElementById("dirs-view");
const labV = document.getElementById("lab-view");
const tEl = document.getElementById("title");
const pEl = document.getElementById("prog");
const video = document.getElementById("video");
const toast = document.getElementById("toast");
function bleep(m){toast.textContent=m;toast.classList.add("show");clearTimeout(bleep._);bleep._=setTimeout(()=>toast.classList.remove("show"),1200);}
async function loadDirs(){
  dirsV.hidden=false; labV.hidden=true; pathEl.textContent="";
  const ds = await fetch("/api/dirs").then(r=>r.json());
  if(!ds.length){dirsV.innerHTML='<div class="empty">No subdirs in root.</div>';return;}
  dirsV.innerHTML='<h3 style="margin:0 0 12px;font-size:14px;color:var(--muted)">Click a language to start</h3><div class="dirs">'+ds.map(d=>
    \`<a class="tile" href="?dir=\${encodeURIComponent(d.lang)}"><div class="lang">\${d.lang}</div><div class="count">\${d.count} files</div></a>\`).join("")+'</div>';
}
let files=[], idx=0;
async function loadDir(d){
  dirsV.hidden=true; labV.hidden=false; pathEl.textContent=d+"/";
  files = await fetch("/api/dir/"+encodeURIComponent(d)).then(r=>r.json());
  idx=0; render();
}
function render(){
  if(idx<0)idx=0;
  if(idx>=files.length){tEl.textContent="✓ done";pEl.textContent=\`\${files.length}/\${files.length}\`;video.removeAttribute("src");bleep("all done in this dir");return;}
  tEl.textContent=files[idx]; pEl.textContent=\`\${idx+1}/\${files.length}\`;
  video.src="/video/"+encodeURIComponent(dir)+"/"+encodeURIComponent(files[idx]); video.load();
}
async function act(a){
  if(a==="prev"){idx=Math.max(0,idx-1);render();return;}
  if(a==="next"||a==="skip"){idx+=1;render();return;}
  let tl = a==="other" ? (prompt("target language code (e.g. de, fr, nn)?")||"").trim().toLowerCase() : a;
  if(!/^[a-z_]{2,10}$/.test(tl)){bleep("invalid code");return;}
  if(tl===dir){bleep("already here");idx+=1;render();return;}
  const fn=files[idx];
  const r=await fetch("/api/move",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({from_lang:dir,to_lang:tl,filename:fn})});
  if(!r.ok){bleep("move failed: "+r.status);return;}
  bleep("→ "+tl+"/");
  files.splice(idx,1); render();
}
document.querySelectorAll("[data-a]").forEach(b=>b.addEventListener("click",()=>act(b.dataset.a)));
document.addEventListener("keydown",e=>{
  if(["INPUT","TEXTAREA"].includes(e.target.tagName))return;
  const m={"1":"sv","2":"en","3":"skip","4":"mixed","m":"mixed","n":"next","p":"prev","o":"other"}[e.key.toLowerCase()];
  if(m){e.preventDefault(); act(m);}
});
if(dir) loadDir(dir); else loadDirs();
</script></body></html>`;
