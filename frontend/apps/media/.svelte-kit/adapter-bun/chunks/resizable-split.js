import "./async.js";
import { s as sanitize_props, a as spread_props, b as slot, i as attr_class, f as clsx, l as ensure_array_like, h as attr, m as escape_html, d as derived, k as stringify, n as attr_style } from "./renderer.js";
import { I as Icon } from "./Icon.js";
import "@sveltejs/kit/internal";
import "./exports.js";
import "./utils.js";
import "@sveltejs/kit/internal/server";
import "./root.js";
import "./state.svelte.js";
import { c as activeView, m as mediaUrl } from "./api.js";
import { J as queryTerms, G as fmtTime, F as hitKey } from "./scroll-lock.js";
import "clsx";
import { C as Chevron_right } from "./sr-only-styles.js";
import { r as run } from "./render-context.js";
function Loader_circle($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [["path", { "d": "M21 12a9 9 0 1 1-6.219-8.56" }]];
  Icon($$renderer, spread_props([
    { name: "loader-circle" },
    $$sanitized_props,
    {
      /**
       * @component @name LoaderCircle
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMjEgMTJhOSA5IDAgMSAxLTYuMjE5LTguNTYiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/loader-circle
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Maximize_2($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["polyline", { "points": "15 3 21 3 21 9" }],
    ["polyline", { "points": "9 21 3 21 3 15" }],
    ["line", { "x1": "21", "x2": "14", "y1": "3", "y2": "10" }],
    ["line", { "x1": "3", "x2": "10", "y1": "21", "y2": "14" }]
  ];
  Icon($$renderer, spread_props([
    { name: "maximize-2" },
    $$sanitized_props,
    {
      /**
       * @component @name Maximize2
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cG9seWxpbmUgcG9pbnRzPSIxNSAzIDIxIDMgMjEgOSIgLz4KICA8cG9seWxpbmUgcG9pbnRzPSI5IDIxIDMgMjEgMyAxNSIgLz4KICA8bGluZSB4MT0iMjEiIHgyPSIxNCIgeTE9IjMiIHkyPSIxMCIgLz4KICA8bGluZSB4MT0iMyIgeDI9IjEwIiB5MT0iMjEiIHkyPSIxNCIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/maximize-2
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Play($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [["polygon", { "points": "6 3 20 12 6 21 6 3" }]];
  Icon($$renderer, spread_props([
    { name: "play" },
    $$sanitized_props,
    {
      /**
       * @component @name Play
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cG9seWdvbiBwb2ludHM9IjYgMyAyMCAxMiA2IDIxIDYgMyIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/play
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Plus($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [["path", { "d": "M5 12h14" }], ["path", { "d": "M12 5v14" }]];
  Icon($$renderer, spread_props([
    { name: "plus" },
    $$sanitized_props,
    {
      /**
       * @component @name Plus
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNNSAxMmgxNCIgLz4KICA8cGF0aCBkPSJNMTIgNXYxNCIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/plus
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function X($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M18 6 6 18" }],
    ["path", { "d": "m6 6 12 12" }]
  ];
  Icon($$renderer, spread_props([
    { name: "x" },
    $$sanitized_props,
    {
      /**
       * @component @name X
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTggNiA2IDE4IiAvPgogIDxwYXRoIGQ9Im02IDYgMTIgMTIiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/x
       * @see https://lucide.dev/guide/packages/lucide-svelte - Documentation
       *
       * @param {Object} props - Lucide icons props and any valid SVG attribute
       * @returns {FunctionalComponent} Svelte component
       *
       */
      iconNode,
      children: ($$renderer2) => {
        $$renderer2.push(`<!--[-->`);
        slot($$renderer2, $$props, "default", {});
        $$renderer2.push(`<!--]-->`);
      },
      $$slots: { default: true }
    }
  ]));
}
function Transcript_highlighter($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { alignments, query = "", chrome = true } = $$props;
    const terms = derived(() => new Set(queryTerms(query)));
    const normalize = (w) => w.replace(/^\W+|\W+$/gu, "").toLowerCase();
    $$renderer2.push(`<div${attr_class(clsx(
      /**
       * Karaoke cursor.
       *
       * The DOM (`scrollContainer`) and the data (`alignments`) are independent
       * reactive sources, so we drive the RAF loop from `$effect`. Whenever
       * either changes, the previous RAF + listeners are torn down via the
       * cleanup return, then the wordMap/sentMap are rebuilt against the
       * current DOM. This was the bug in my first port — `{@attach}` only
       * runs once per parent <div> mount, but the inner spans re-render on
       * every hit change, so the captured maps went stale.
       */
      // Snapshot the element for this run. `media` is a prop (a live getter), so
      // reading it in cleanup could see a newer/null value and detach from the
      // wrong element (or null-deref). Capture once → add, remove, and the RAF
      // tick all act on the same element. Mirrors player-pane's `const el`.
      // Touch `alignments` so the effect re-runs when a new hit is selected
      // and the spans below have been replaced by Svelte's reconciler.
      // `timeupdate` fires ~4–66×/s during playback (enough for word-level
      // karaoke) and costs NOTHING when paused, while `seeked` covers dragging
      // the playhead on a paused video. This replaces the old always-on
      // requestAnimationFrame(tick) loop — up to 3 windowed highlighters share
      // one media element, so the RAF loops were 3 idle 60fps spinners on a
      // paused video for zero benefit over these two listeners.
      chrome ? "rounded-md border border-border bg-surface2 p-3 text-sm leading-7" : "p-3"
    ))}><!--[-->`);
    const each_array = ensure_array_like(alignments);
    for (let $$index_1 = 0, $$length = each_array.length; $$index_1 < $$length; $$index_1++) {
      let a = each_array[$$index_1];
      const sentEndsWithSpace = (a.text ?? "").endsWith(" ");
      $$renderer2.push(`<span data-sentence=""${attr("data-start", a.start)}${attr("data-end", a.end)} class="rounded-sm transition-colors hover:bg-secondary/40 cursor-pointer" role="button" tabindex="0"><!--[-->`);
      const each_array_1 = ensure_array_like(a.words ?? []);
      for (let $$index = 0, $$length2 = each_array_1.length; $$index < $$length2; $$index++) {
        let w = each_array_1[$$index];
        const stripped = normalize(w.text ?? "");
        $$renderer2.push(`<span data-word=""${attr("data-start", w.start)}${attr("data-end", w.end)}${attr_class("rounded-sm", void 0, {
          "underline": terms().has(stripped),
          "decoration-highlight": terms().has(stripped),
          "decoration-2": terms().has(stripped),
          "underline-offset-2": terms().has(stripped)
        })}>${escape_html(w.text)}</span>`);
      }
      $$renderer2.push(`<!--]--></span> `);
      if (!sentEndsWithSpace) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<br/>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function Transcript_window($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      chunks,
      currentChunkIdx,
      windowStartIdx,
      query = "",
      variant = "panel"
    } = $$props;
    const blockClass = (isCurrent) => {
      const base = "border-l-2 px-3 py-1";
      if (variant === "overlay") {
        return isCurrent ? `${base} border-primary bg-white/5 text-white` : `${base} border-transparent text-white/80 opacity-50`;
      }
      return isCurrent ? `${base} border-primary bg-primary/5` : `${base} border-transparent text-muted-foreground opacity-60`;
    };
    $$renderer2.push(`<div${attr_class(clsx(variant === "overlay" ? "text-left" : ""))}><!--[-->`);
    const each_array = ensure_array_like(chunks);
    for (let j = 0, $$length = each_array.length; j < $$length; j++) {
      let c = each_array[j];
      const isCurrent = windowStartIdx + j === currentChunkIdx;
      const start = activeView().time(c)?.start ?? null;
      $$renderer2.push(`<div${attr_class(clsx(blockClass(isCurrent)))}><div class="flex items-center gap-1.5 py-0.5 text-[11px] font-mono text-muted-foreground">`);
      if (start != null) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<span>${escape_html(fmtTime(start))}</span>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (isCurrent) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<span class="inline-flex items-center gap-0.5 font-sans font-medium text-primary">`);
        Play($$renderer2, { class: "size-3 fill-current" });
        $$renderer2.push(`<!---->playing</span>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></div> `);
      Transcript_highlighter($$renderer2, { alignments: c.alignments ?? [], query, chrome: false });
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function Chunk_timeline($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    activeView();
    {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function Player_pane($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { hit, query = "" } = $$props;
    const WINDOW_BEFORE = 1;
    const WINDOW_AFTER = 1;
    let currentTime = 0;
    let docChunks = [];
    let tab = "transcript";
    const currentChunkIdx = derived(() => {
      const cs = docChunks;
      if (cs.length === 0) return 0;
      const view = activeView();
      const t = currentTime;
      const i = cs.findIndex((c) => {
        const span = view.time(c);
        return span != null && t >= span.start && t < span.end;
      });
      if (i !== -1) return i;
      const h = hit;
      if (h) {
        const hk = hitKey(h);
        const j = cs.findIndex((c) => hitKey(c) === hk);
        if (j !== -1) return j;
      }
      return 0;
    });
    const windowStartIdx = derived(() => Math.max(0, currentChunkIdx() - WINDOW_BEFORE));
    const windowChunks = derived(() => {
      const cs = docChunks;
      if (cs.length === 0) return [];
      const lo = windowStartIdx();
      const hi = Math.min(cs.length, currentChunkIdx() + WINDOW_AFTER + 1);
      return cs.slice(lo, hi);
    });
    const activeKey = derived(() => hit ? hitKey(hit) : null);
    let showMeta = false;
    const metaRows = derived(() => {
      const h = hit;
      if (!h) return [];
      const view = activeView();
      const rows = [];
      const caption = view.caption(h);
      if (caption) rows.push(["Caption", caption]);
      const t = view.time(h);
      if (t) rows.push(["Time", `${fmtTime(t.start)} → ${fmtTime(t.end)}`]);
      const dur = view.duration(h);
      if (dur != null) rows.push(["Length", fmtTime(dur)]);
      for (const m of view.metadata(h)) rows.push([m.label, m.value]);
      return rows;
    });
    $$renderer2.push(`<div class="flex h-full min-h-0 flex-col gap-3 p-4">`);
    if (
      /**
      * Whenever `hit` changes, seek the player to the hit's start and play.
      *
      * `src` is owned reactively by the `<video src={mediaUrl(hit)}>` binding
      * below — this effect must NOT touch it. The old code called
      * `el.removeAttribute('src'); el.load()` in cleanup; when the next hit was
      * in the *same* document the bound URL didn't change, so Svelte never
      * re-applied `src` and the element was left sourceless and wedged (the
      * "second click freezes the player until full refresh" bug).
      *
      * We read the hit's doc id + start so the effect re-runs on either a new
      * document (src changes → metadata reloads → seek on `loadedmetadata`) or a
      * new chunk in the same already-loaded document (seek immediately, since
      * `loadedmetadata`/`canplay` won't fire again).
      */
      // ignore — some browsers throw if metadata isn't fully ready
      /* HAVE_METADATA */
      !hit
    ) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="m-auto text-sm text-muted-foreground">Click a hit to play.</div>`);
    } else {
      let fsToggle = function($$renderer3, cls) {
        $$renderer3.push(`<button type="button"${attr("title", "Fullscreen with transcript")}${attr("aria-label", "Fullscreen with transcript")}${attr_class(clsx(cls))}>`);
        {
          $$renderer3.push("<!--[-1-->");
          Maximize_2($$renderer3, { class: "size-4" });
        }
        $$renderer3.push(`<!--]--></button>`);
      };
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div${attr_class(clsx("relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-border bg-card shadow-sm"))}>`);
      {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--> <video controls="" controlslist="nofullscreen" preload="metadata"${attr("src", mediaUrl(hit))}${attr_class(clsx("aspect-video max-h-[45vh] w-full shrink-0 bg-black object-contain"))}><track kind="captions"/></video> `);
      {
        $$renderer2.push("<!--[0-->");
        Chunk_timeline($$renderer2, {
          activeKey: activeKey(),
          currentChunkIdx: currentChunkIdx()
        });
        $$renderer2.push(`<!----> <div class="flex items-center gap-1.5 border-t border-border/70 px-3 py-1.5 text-xs font-medium text-muted-foreground"><button type="button"${attr("aria-pressed", tab === "transcript")}${attr_class(`rounded px-2 py-0.5 transition-colors ${stringify(
          "bg-secondary text-foreground"
        )}`)}>Transcript</button> <button type="button"${attr("aria-pressed", tab === "speakers")}${attr_class(`rounded px-2 py-0.5 transition-colors ${stringify("hover:bg-secondary hover:text-foreground")}`)}>Speakers</button> `);
        fsToggle($$renderer2, "ml-auto rounded p-1 transition-colors hover:bg-secondary hover:text-foreground");
        $$renderer2.push(`<!----></div>`);
      }
      $$renderer2.push(`<!--]--> <div${attr_class(clsx("min-h-[8rem] flex-1 overflow-y-auto text-sm leading-7"))}>`);
      {
        $$renderer2.push("<!--[-1-->");
        Transcript_window($$renderer2, {
          chunks: windowChunks(),
          currentChunkIdx: currentChunkIdx(),
          windowStartIdx: windowStartIdx(),
          query,
          variant: "panel"
        });
      }
      $$renderer2.push(`<!--]--></div></div> `);
      {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (metaRows().length) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<div class="shrink-0 overflow-hidden rounded-md border border-border bg-card/40 text-xs"><button type="button"${attr("aria-expanded", showMeta)} class="flex w-full items-center gap-1.5 px-3 py-1.5 text-muted-foreground transition-colors hover:bg-secondary/40">`);
        Chevron_right($$renderer2, {
          class: `size-3.5 transition-transform ${stringify("")}`
        });
        $$renderer2.push(`<!----> <span class="font-medium">Metadata</span> <span class="ml-auto text-[10px] text-muted-foreground/70">${escape_html(metaRows().length)} fields</span></button> `);
        {
          $$renderer2.push("<!--[-1-->");
        }
        $$renderer2.push(`<!--]--></div>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function Resizable_split($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      left,
      right,
      storageKey = "lance-media-split",
      initial = 0.6,
      minLeft = 360,
      minRight = 320,
      orientation = "horizontal"
    } = $$props;
    const vertical = derived(() => orientation === "vertical");
    let fraction = run(() => initial);
    let dragging = false;
    $$renderer2.push(`<div${attr_class("grid h-full min-h-0", void 0, {
      "cursor-col-resize": dragging,
      "cursor-row-resize": dragging,
      "select-none": dragging
    })}${attr_style("", {
      "grid-template-columns": vertical() ? void 0 : `${(fraction * 100).toFixed(2)}% 6px 1fr`,
      "grid-template-rows": vertical() ? `${(fraction * 100).toFixed(2)}% 6px 1fr` : void 0
    })}><div class="min-h-0 overflow-hidden">`);
    left($$renderer2);
    $$renderer2.push(`<!----></div> <button type="button" aria-label="Resize panels" title="Drag to resize · double-click to reset"${attr_class("group relative flex items-center justify-center border-border bg-secondary/40 transition-colors hover:bg-primary/30 active:bg-primary/40 focus-visible:bg-primary/40 focus-visible:outline-none", void 0, {
      "cursor-col-resize": (
        // Same px clamps as the pointer path (0.2/0.8 ignored minLeft/minRight
        // and could violate them), and persist like a drag does.
        !vertical()
      ),
      "cursor-row-resize": vertical(),
      "border-x": !vertical(),
      "border-y": vertical()
    })}><span aria-hidden="true"${attr_class("flex gap-0.5 text-muted-foreground/70 group-hover:text-foreground", void 0, { "flex-col": !vertical(), "flex-row": vertical() })}><span class="size-0.5 rounded-full bg-current"></span> <span class="size-0.5 rounded-full bg-current"></span> <span class="size-0.5 rounded-full bg-current"></span> <span class="size-0.5 rounded-full bg-current"></span> <span class="size-0.5 rounded-full bg-current"></span></span> <span class="pointer-events-none absolute top-1/2 left-1/2 -translate-x-1/2 translate-y-6 whitespace-nowrap rounded border border-border bg-card px-2 py-0.5 text-[10px] text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100">drag to resize</span></button> <div class="min-h-0 overflow-hidden">`);
    right($$renderer2);
    $$renderer2.push(`<!----></div></div>`);
  });
}
export {
  Loader_circle as L,
  Plus as P,
  Resizable_split as R,
  X,
  Play as a,
  Player_pane as b
};
