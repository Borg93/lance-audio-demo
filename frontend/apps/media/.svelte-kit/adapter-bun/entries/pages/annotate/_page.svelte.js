import "../../../chunks/async.js";
import { m as escape_html, l as ensure_array_like, i as attr_class, k as stringify, h as attr, c as setContext, j as bind_props } from "../../../chunks/renderer.js";
import "clsx";
import { tableFromIPC } from "apache-arrow";
import "pixi.js";
function AnnotationTable($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let rows = [];
    const statusColor = (s) => s === "prediction" ? "text-amber-600" : s === "accepted" ? "text-emerald-600" : s === "rejected" ? "text-red-600" : "text-muted-foreground";
    $$renderer2.push(`<div class="flex h-full flex-col overflow-hidden text-xs" data-testid="annotation-table"><div class="border-b border-border px-3 py-2 font-medium">Review queue · ${escape_html(rows.length)} annotation${escape_html(rows.length === 1 ? "" : "s")}</div> `);
    {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div class="overflow-auto"><table class="w-full border-collapse"><thead class="sticky top-0 bg-muted/60 text-left text-[10px] uppercase text-muted-foreground"><tr><th class="px-3 py-1.5">Label</th><th class="px-2 py-1.5">Shape</th><th class="px-2 py-1.5">Status</th><th class="px-2 py-1.5 text-right">Conf</th><th class="px-2 py-1.5 text-right">Uncert</th><th class="px-3 py-1.5">Text</th></tr></thead><tbody><!--[-->`);
      const each_array = ensure_array_like(rows);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let row = each_array[$$index];
        $$renderer2.push(`<tr class="cursor-pointer border-b border-border/50 hover:bg-muted/40"><td class="px-3 py-1.5">${escape_html(row.label || "—")}</td><td class="px-2 py-1.5 text-muted-foreground">${escape_html(row.shape_type)}</td><td${attr_class(`px-2 py-1.5 font-medium ${stringify(statusColor(row.status))}`)}>${escape_html(row.status)}</td><td class="px-2 py-1.5 text-right tabular-nums">${escape_html(row.confidence.toFixed(2))}</td><td class="px-2 py-1.5 text-right tabular-nums">${escape_html(row.uncertainty.toFixed(2))}</td><td class="max-w-[16rem] truncate px-3 py-1.5"${attr("title", row.text)}>${escape_html(row.text || "—")}</td></tr>`);
      }
      $$renderer2.push(`<!--]--></tbody></table></div>`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function AudioViewer($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { unit, onload } = $$props;
    $$renderer2.push(`<div class="flex h-full w-full items-center justify-center p-6 text-center text-sm text-muted-foreground"><div><div class="font-medium">audio annotation — scaffold</div> <div class="mt-1 text-xs">peaks.js waveform · temporal segments (t_start/t_end) · ${escape_html(unit.key)}</div></div></div>`);
  });
}
function PixiCanvas($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      children,
      zoom = 1,
      panX = 0,
      panY = 0,
      colorFn,
      annotationStyle,
      onready
    } = $$props;
    const pixiCtx = {
      app: null,
      plugins: { image: null, arrow: null, interaction: null }
    };
    setContext("pixi", pixiCtx);
    $$renderer2.push(`<div class="pixi-grid relative h-full w-full overflow-hidden" style="cursor: default;">`);
    {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, { zoom, panX, panY });
  });
}
function ImageViewer($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { unit, onload } = $$props;
    async function onready(ctx) {
      if (unit.imageUrl) await ctx.plugins.image.load(unit.imageUrl);
      const res = await fetch(unit.annotationsUrl);
      if (!res.ok) throw new Error(`annotations HTTP ${res.status}`);
      const table = tableFromIPC(new Uint8Array(await res.arrayBuffer()));
      ctx.plugins.arrow.load(table);
      ctx.plugins.arrow.sync();
      onload?.(table.numRows);
    }
    PixiCanvas($$renderer2, { onready });
  });
}
function VideoViewer($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { unit, onload } = $$props;
    $$renderer2.push(`<div class="flex h-full w-full items-center justify-center p-6 text-center text-sm text-muted-foreground"><div><div class="font-medium">video annotation — scaffold</div> <div class="mt-1 text-xs">frame overlay: mask · pixels · bbox/polygon (@currentTime) + audio-track timeline · ${escape_html(unit.key)}</div></div></div>`);
  });
}
function mediaKindOf(mime) {
  const m = mime.toLowerCase();
  if (m.startsWith("video/")) return "video";
  if (m.startsWith("audio/")) return "audio";
  return "image";
}
const VIEWERS = {
  image: ImageViewer,
  audio: AudioViewer,
  video: VideoViewer
};
function viewerFor(kind) {
  return VIEWERS[kind];
}
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const KEY = "fe00cd746463ad2c/0/19";
    const unit = {
      kind: mediaKindOf("image/jpeg"),
      key: KEY,
      imageUrl: `/api/chunk-frame/${KEY}`,
      annotationsUrl: `/api/annotations/${KEY}`
    };
    const Viewer = viewerFor(unit.kind);
    let status = "loading…";
    $$renderer2.push(`<div class="flex h-screen w-screen"><div class="relative flex-1"><div class="absolute left-3 top-3 z-10 rounded bg-black/70 px-2 py-1 font-mono text-xs text-white" data-testid="annotate-status">annotate · ${escape_html(unit.kind)} · ${escape_html(status)}</div> `);
    Viewer($$renderer2, { unit, onload: (n) => status = `${n} annotations from Lance` });
    $$renderer2.push(`<!----></div> <aside class="w-[28rem] shrink-0 border-l border-border bg-card">`);
    AnnotationTable($$renderer2);
    $$renderer2.push(`<!----></aside></div>`);
  });
}
export {
  _page as default
};
