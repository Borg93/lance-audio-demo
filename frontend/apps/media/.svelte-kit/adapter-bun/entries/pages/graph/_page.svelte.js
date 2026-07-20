import "../../../chunks/async.js";
import { s as sanitize_props, a as spread_props, b as slot, l as ensure_array_like, m as escape_html, d as derived, h as attr, n as attr_style, k as stringify } from "../../../chunks/renderer.js";
import "@sveltejs/kit/internal";
import "../../../chunks/exports.js";
import "../../../chunks/utils.js";
import "@sveltejs/kit/internal/server";
import "../../../chunks/root.js";
import "../../../chunks/state.svelte.js";
import { forceSimulation, forceManyBody, forceLink, forceCollide, forceX, forceY } from "d3-force";
import { B as Button } from "../../../chunks/scroll-lock.js";
import "clsx";
import "../../../chunks/descriptor.js";
import { S as Select_1 } from "../../../chunks/select.js";
import { C as Chevron_right } from "../../../chunks/sr-only-styles.js";
import { I as Icon } from "../../../chunks/Icon.js";
import { h as getGraphSubgraph, j as runGraphCypher } from "../../../chunks/api.js";
function Ellipsis($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["circle", { "cx": "12", "cy": "12", "r": "1" }],
    ["circle", { "cx": "19", "cy": "12", "r": "1" }],
    ["circle", { "cx": "5", "cy": "12", "r": "1" }]
  ];
  Icon($$renderer, spread_props([
    { name: "ellipsis" },
    $$sanitized_props,
    {
      /**
       * @component @name Ellipsis
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxIiAvPgogIDxjaXJjbGUgY3g9IjE5IiBjeT0iMTIiIHI9IjEiIC8+CiAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIxIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/ellipsis
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
function Graph_breadcrumb($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { trail } = $$props;
    const MAX_VISIBLE = 3;
    let expandedFor = null;
    const expanded = derived(() => expandedFor === trail);
    const hidden = derived(() => expanded() || trail.length <= MAX_VISIBLE ? [] : trail.slice(0, -2));
    const visible = derived(() => expanded() || trail.length <= MAX_VISIBLE ? trail : trail.slice(-2));
    $$renderer2.push(`<nav aria-label="breadcrumb" class="min-w-0"><ol class="text-muted-foreground flex flex-wrap items-center gap-1 text-sm"><li>`);
    if (trail.length === 0) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<span aria-current="page" class="text-foreground font-medium">Overview</span>`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<button class="hover:text-foreground transition-colors">Overview</button>`);
    }
    $$renderer2.push(`<!--]--></li> `);
    if (hidden().length > 0) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<li aria-hidden="true">`);
      Chevron_right($$renderer2, { class: "size-3.5" });
      $$renderer2.push(`<!----></li> <li><button class="hover:text-foreground flex items-center transition-colors" aria-label="Show full path">`);
      Ellipsis($$renderer2, { class: "size-4" });
      $$renderer2.push(`<!----></button></li>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--> <!--[-->`);
    const each_array = ensure_array_like(visible());
    for (let i = 0, $$length = each_array.length; i < $$length; i++) {
      let crumb = each_array[i];
      $$renderer2.push(`<li aria-hidden="true">`);
      Chevron_right($$renderer2, { class: "size-3.5" });
      $$renderer2.push(`<!----></li> <li class="min-w-0">`);
      if (i === visible().length - 1) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<span aria-current="page" class="text-foreground block max-w-48 truncate font-medium">${escape_html(crumb.name)}</span>`);
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`<button class="hover:text-foreground block max-w-40 truncate transition-colors">${escape_html(crumb.name)}</button>`);
      }
      $$renderer2.push(`<!--]--></li>`);
    }
    $$renderer2.push(`<!--]--></ol></nav>`);
  });
}
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const TYPE_RGB = {
      PERSON: [56, 189, 248],
      // sky-400
      ORG: [52, 211, 153],
      // emerald-400
      GEO: [251, 191, 36],
      // amber-400
      EVENT: [167, 139, 250],
      // violet-400
      CONCEPT: [244, 114, 182],
      // pink-400 — ideas / policy areas / methods
      WORK: [251, 146, 60],
      // orange-400 — reports / artifacts / datasets
      OTHER: [148, 163, 184]
      // slate-400
    };
    function nodeColor(type, videos) {
      const base = TYPE_RGB[type] ?? TYPE_RGB.OTHER;
      if (videos <= 1) return base;
      return [
        Math.round(base[0] + (255 - base[0]) * 0.35),
        Math.round(base[1] + (255 - base[1]) * 0.35),
        Math.round(base[2] + (255 - base[2]) * 0.35)
      ];
    }
    const sizeOf = (mentions) => 8 + Math.min(20, Math.sqrt(Math.max(1, mentions)) * 4);
    const PRESETS = [
      { value: "", label: "Cypher presets…" },
      {
        value: "MATCH (a:Entity)-[:MENTIONS]->(c:Chunk)\nRETURN a.name, c.doc_id, c.start_s LIMIT 25",
        label: "Entity → its clips"
      },
      {
        value: "MATCH (a:Entity)-[:MENTIONS]->(c:Chunk)<-[:MENTIONS]-(b:Entity)\nWHERE a.entity_id < b.entity_id\nRETURN a.name, b.name, count(c.chunk_id) AS shared ORDER BY shared DESC LIMIT 15",
        label: "Co-occurrence (shared clips)"
      },
      {
        value: "MATCH (a:Entity)-[:RELATIONSHIP]->(b:Entity)\nRETURN a.name, b.name LIMIT 25",
        label: "Relationship neighbourhood"
      },
      {
        value: "MATCH (a:Entity)\nRETURN a.name, a.entity_type, a.mention_count ORDER BY a.mention_count DESC LIMIT 20",
        label: "Top entities by mentions"
      },
      {
        value: "MATCH (a:Entity)\nRETURN a.entity_type, count(a.entity_id) AS n ORDER BY n DESC LIMIT 10",
        label: "Entity-type breakdown"
      }
    ];
    const HELP_EXAMPLES = [
      {
        label: "Who are the people?",
        desc: "Named individuals only (full names), ranked by how many clips mention them.",
        query: "MATCH (a:Entity) WHERE a.entity_type = 'PERSON' AND a.name CONTAINS ' ' RETURN a.name, a.mention_count ORDER BY a.mention_count DESC LIMIT 20"
      },
      {
        label: "What were the topics?",
        desc: "The most-discussed concepts and policy areas across every press conference.",
        query: "MATCH (a:Entity) WHERE a.entity_type = 'CONCEPT' RETURN a.name, a.mention_count ORDER BY a.mention_count DESC LIMIT 20"
      },
      {
        label: "Central players (hubs)",
        desc: "Entities with the most connections — the gravitational centres of the whole conversation.",
        query: "MATCH (a:Entity)-[:RELATIONSHIP]->(b:Entity) RETURN a.name, a.entity_type, count(b.entity_id) AS connections ORDER BY connections DESC LIMIT 15"
      },
      {
        label: "Strongest links",
        desc: "Relationships asserted most often (weight = shared clips), e.g. Sweden↔EU. Each row carries the relation text.",
        query: "MATCH (a:Entity)-[r:RELATIONSHIP]->(b:Entity) RETURN a.name, b.name, r.weight, r.description ORDER BY r.weight DESC LIMIT 15"
      },
      {
        label: "One person's network",
        desc: "Everything Göran Persson is linked to. Swap the name (lower-case) to explore anyone.",
        query: "MATCH (a:Entity)-[:RELATIONSHIP]-(b:Entity) WHERE a.name_lower = 'göran persson' RETURN b.name, b.entity_type LIMIT 25"
      },
      {
        label: "Who comes up with the EU?",
        desc: "Organisations most often mentioned in the same clips as the EU (co-occurrence).",
        query: "MATCH (a:Entity)-[:MENTIONS]->(c:Chunk)<-[:MENTIONS]-(b:Entity) WHERE a.name_lower = 'eu' AND b.entity_type = 'ORG' AND b.name_lower <> 'eu' RETURN b.name, count(c.chunk_id) AS shared ORDER BY shared DESC LIMIT 12"
      },
      {
        label: "A person on the timeline",
        desc: "Every clip mentioning Carl Bildt, in order — click a result row to jump to that moment in the video.",
        query: "MATCH (a:Entity)-[:MENTIONS]->(c:Chunk) WHERE a.name_lower = 'carl bildt' RETURN c.doc_id, c.start_s, c.text ORDER BY c.start_s LIMIT 20"
      },
      {
        label: "Topics tied to a place",
        desc: "Which concepts come up alongside Sweden — what the country is discussed in terms of.",
        query: "MATCH (a:Entity)-[:MENTIONS]->(c:Chunk)<-[:MENTIONS]-(b:Entity) WHERE a.name_lower = 'sverige' AND b.entity_type = 'CONCEPT' RETURN b.name, count(c.chunk_id) AS shared ORDER BY shared DESC LIMIT 15"
      },
      {
        label: "Search by word",
        desc: "Find any entity whose name contains a word (try miljö, skatt, vård…).",
        query: "MATCH (a:Entity) WHERE a.name_lower CONTAINS 'miljö' RETURN a.name, a.entity_type, a.mention_count LIMIT 20"
      }
    ];
    let view = "graph";
    let loading = false;
    let showHelp = false;
    let searchQ = "";
    let presetValue = "";
    let cypherText = "";
    let cypherOpen = false;
    let cypherResult = null;
    let detail = null;
    let selectedId = null;
    let trail = [];
    let graphNodes = [];
    let nodesX = new Float32Array(0);
    let nodesY = new Float32Array(0);
    let sim = null;
    let simNodes = [];
    function stopSim() {
      sim?.stop();
      sim = null;
    }
    function recomputeFit() {
      const n = nodesX.length;
      if (n === 0) return;
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      for (let i = 0; i < n; i++) {
        minX = Math.min(minX, nodesX[i]);
        minY = Math.min(minY, nodesY[i]);
        maxX = Math.max(maxX, nodesX[i]);
        maxY = Math.max(maxY, nodesY[i]);
      }
    }
    function buildLayout(nodes, edges) {
      stopSim();
      const n = nodes.length;
      const idx = new Map(nodes.map((nd, i) => [nd.id, i]));
      const x = new Float32Array(n);
      const y = new Float32Array(n);
      const rgb = new Uint8Array(n * 3);
      const size = new Float32Array(n);
      const R = 40 + n * 1.5;
      for (let i = 0; i < n; i++) {
        const a = i / Math.max(1, n) * Math.PI * 2;
        x[i] = Math.cos(a) * R;
        y[i] = Math.sin(a) * R;
        const [r, g, b] = nodeColor(nodes[i].type, nodes[i].videos);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
        size[i] = sizeOf(nodes[i].mentions);
      }
      const valid = edges.filter((e) => idx.has(e.source) && idx.has(e.target));
      const ei = new Uint32Array(valid.length * 2);
      valid.forEach((e, i) => {
        ei[i * 2] = idx.get(e.source);
        ei[i * 2 + 1] = idx.get(e.target);
      });
      graphNodes = nodes;
      nodesX = x;
      nodesY = y;
      simNodes = nodes.map((nd, i) => ({ id: nd.id, idx: i, x: x[i], y: y[i] }));
      const links = valid.map((e) => ({ source: e.source, target: e.target }));
      sim = forceSimulation(simNodes).stop().force("charge", forceManyBody().strength(-220)).force("link", forceLink(links).id((d) => d.id).distance(46).strength(0.35)).force("collide", forceCollide((d) => sizeOf(graphNodes[d.idx]?.mentions ?? 1) / 2 + 5)).force("x", forceX(0).strength(0.04)).force("y", forceY(0).strength(0.04));
      for (let tick = 0; sim.alpha() > 0.025 && tick < 400; tick++) sim.tick();
      for (let i = 0; i < simNodes.length; i++) {
        nodesX[i] = simNodes[i].x ?? 0;
        nodesY[i] = simNodes[i].y ?? 0;
      }
      recomputeFit();
    }
    async function loadOverview() {
      loading = true;
      try {
        const sub = await getGraphSubgraph(void 0, 120);
        selectedId = null;
        detail = null;
        trail = [];
        buildLayout(sub.nodes, sub.edges);
      } finally {
        loading = false;
      }
    }
    async function runCypher() {
      const q = cypherText.trim();
      if (!q) return;
      cypherOpen = true;
      view = "table";
      cypherResult = await runGraphCypher(q);
    }
    const fmtTime = (s) => {
      const m = Math.floor(s / 60);
      return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
    };
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="flex h-full min-h-0"><div class="flex min-w-0 flex-1 flex-col"><div class="border-border bg-card/40 flex flex-col gap-2 border-b p-3"><div class="flex flex-wrap items-center gap-2"><div class="relative w-64"><input class="border-border bg-background focus-visible:ring-ring h-8 w-full rounded-md border px-3 text-sm focus-visible:ring-2 focus-visible:outline-none" placeholder="Search entities…"${attr("value", searchQ)}/> `);
      {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--></div> <div class="w-56">`);
      Select_1($$renderer3, {
        options: PRESETS,
        placeholder: "Cypher presets…",
        ariaLabel: "Cypher presets",
        get value() {
          return presetValue;
        },
        set value($$value) {
          presetValue = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----></div> `);
      Button($$renderer3, {
        variant: "outline",
        size: "sm",
        onclick: () => cypherOpen = !cypherOpen,
        children: ($$renderer4) => {
          $$renderer4.push(`<!---->${escape_html(cypherOpen ? "Hide" : "Cypher")}`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----> `);
      Button($$renderer3, {
        variant: "outline",
        size: "sm",
        onclick: loadOverview,
        children: ($$renderer4) => {
          $$renderer4.push(`<!---->Overview`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----> <div class="ml-auto flex items-center gap-1">`);
      Button($$renderer3, {
        variant: showHelp ? "default" : "ghost",
        size: "sm",
        onclick: () => showHelp = !showHelp,
        title: "How to read this",
        children: ($$renderer4) => {
          $$renderer4.push(`<!---->?`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----> <div class="bg-border mx-1 h-4 w-px"></div> <!--[-->`);
      const each_array_1 = ensure_array_like(["graph", "table", "json"]);
      for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
        let v = each_array_1[$$index_1];
        Button($$renderer3, {
          variant: view === v ? "default" : "ghost",
          size: "sm",
          onclick: () => view = v,
          class: "capitalize",
          children: ($$renderer4) => {
            $$renderer4.push(`<!---->${escape_html(v)}`);
          },
          $$slots: { default: true }
        });
      }
      $$renderer3.push(`<!--]--></div></div> `);
      Graph_breadcrumb($$renderer3, {
        trail
      });
      $$renderer3.push(`<!----> `);
      if (cypherOpen) {
        $$renderer3.push("<!--[0-->");
        $$renderer3.push(`<div class="flex items-start gap-2"><textarea class="border-border bg-background focus-visible:ring-ring h-20 flex-1 rounded-md border p-2 font-mono text-xs focus-visible:ring-2 focus-visible:outline-none" placeholder="MATCH (a:Entity)-[:RELATIONSHIP]->(b:Entity) RETURN a.name, b.name LIMIT 25">`);
        const $$body = escape_html(cypherText);
        if ($$body) {
          $$renderer3.push(`${$$body}`);
        }
        $$renderer3.push(`</textarea> `);
        Button($$renderer3, {
          size: "sm",
          onclick: runCypher,
          children: ($$renderer4) => {
            $$renderer4.push(`<!---->Run`);
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!----></div>`);
      } else {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (showHelp) {
        $$renderer3.push("<!--[0-->");
        $$renderer3.push(`<div class="bg-muted/40 border-border text-muted-foreground rounded-md border p-3 text-xs leading-relaxed"><p class="text-foreground mb-1 font-medium">What is this?</p> A knowledge graph LightRAG extracted from the press-conference transcripts. Each <b>node is an entity</b> (person, organisation, place, event…) and each <b>edge is a
          relationship</b> it pulled from the text. Everything is queried live with Cypher via
          lance-graph. <p class="text-foreground mt-2 mb-1 font-medium">How to use it</p> <ul class="ml-4 list-disc space-y-0.5"><li><b>Search</b> an entity (top-left) or click a node → its side panel opens.</li> <li>The side panel lists <b>clips</b> (the 30 s moments it's mentioned — click one to play
              it), plus <b>related</b> and <b>co-occurring</b> entities (click to re-centre).</li> <li><b>Cypher presets</b> / the REPL answer precise questions; results show in the <b>Table</b> / <b>JSON</b> tabs.</li></ul> <p class="text-foreground mt-2 mb-1 font-medium">How to read the graph</p> <ul class="ml-4 list-disc space-y-0.5"><li><b>Node size</b> = how many clips mention it; <b>brighter</b> = appears in >1 video.</li> <li><b>Colour</b> = type (legend, bottom-left). Drag to pan, scroll to zoom, “Fit” to recentre.</li> <li>The picture is for <i>navigating</i> — the precise answers live in the clips list and
              the Cypher/Table view.</li> <li><b>One name can be several people.</b> A bare first name like <i>“Anders”</i> is one
              node, but the transcripts may mean different individuals — the AI links them by name
              only. <b>The clips disambiguate</b>: open the node and read each clip's context. <i>Full names</i> (“Göran Persson”) are unambiguous.</li></ul> <p class="mt-1.5">Click an <b>example query</b> in the right-hand rail to run it. Full schema: <code>docs/GRAPH.md</code>.</p></div>`);
      } else {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--></div> <div class="flex min-h-0 flex-1"><div class="relative min-h-0 flex-1">`);
      if (view === "graph") {
        $$renderer3.push("<!--[1-->");
        $$renderer3.push(`<div class="absolute inset-0">`);
        {
          $$renderer3.push("<!--[-1-->");
          $$renderer3.push(`<div class="text-muted-foreground grid h-full place-items-center text-sm">${escape_html(loading ? "Loading graph…" : "No nodes.")}</div>`);
        }
        $$renderer3.push(`<!--]--> `);
        {
          $$renderer3.push("<!--[-1-->");
        }
        $$renderer3.push(`<!--]--> <div class="absolute right-2 bottom-2">`);
        Button($$renderer3, {
          variant: "outline",
          size: "sm",
          onclick: recomputeFit,
          children: ($$renderer4) => {
            $$renderer4.push(`<!---->Fit`);
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!----></div> <div class="bg-card/70 border-border text-muted-foreground absolute bottom-2 left-2 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border px-2 py-1 text-[10px] backdrop-blur"><!--[-->`);
        const each_array_2 = ensure_array_like(Object.entries(TYPE_RGB));
        for (let $$index_2 = 0, $$length = each_array_2.length; $$index_2 < $$length; $$index_2++) {
          let [t, rgb] = each_array_2[$$index_2];
          $$renderer3.push(`<span class="flex items-center gap-1"><span class="size-2 rounded-full"${attr_style("", {
            background: `rgb(${stringify(rgb[0])},${stringify(rgb[1])},${stringify(rgb[2])})`
          })}></span> ${escape_html(t)}</span>`);
        }
        $$renderer3.push(`<!--]--> <span class="text-muted-foreground/70">· size = mentions · click a node for clips</span></div></div>`);
      } else if (view === "table") {
        $$renderer3.push("<!--[2-->");
        $$renderer3.push(`<div class="h-full overflow-auto p-3">`);
        if (cypherResult?.error) {
          $$renderer3.push("<!--[0-->");
          $$renderer3.push(`<pre class="text-destructive border-destructive/40 bg-destructive/10 rounded-md border p-3 text-xs whitespace-pre-wrap">${escape_html(cypherResult.error)}</pre>`);
        } else if (cypherResult && cypherResult.columns.length > 0) {
          $$renderer3.push("<!--[1-->");
          $$renderer3.push(`<table class="w-full text-left text-sm"><thead class="text-muted-foreground border-border border-b"><tr><!--[-->`);
          const each_array_3 = ensure_array_like(cypherResult.columns);
          for (let $$index_3 = 0, $$length = each_array_3.length; $$index_3 < $$length; $$index_3++) {
            let c = each_array_3[$$index_3];
            $$renderer3.push(`<th class="px-2 py-1 font-medium">${escape_html(c)}</th>`);
          }
          $$renderer3.push(`<!--]--></tr></thead><tbody><!--[-->`);
          const each_array_4 = ensure_array_like(cypherResult.rows);
          for (let i = 0, $$length = each_array_4.length; i < $$length; i++) {
            let row = each_array_4[i];
            $$renderer3.push(`<tr class="border-border/50 border-b"><!--[-->`);
            const each_array_5 = ensure_array_like(row);
            for (let j = 0, $$length2 = each_array_5.length; j < $$length2; j++) {
              let cell = each_array_5[j];
              $$renderer3.push(`<td class="px-2 py-1 align-top">${escape_html(cell === null ? "—" : cell)}</td>`);
            }
            $$renderer3.push(`<!--]--></tr>`);
          }
          $$renderer3.push(`<!--]--></tbody></table> <p class="text-muted-foreground mt-2 text-xs">${escape_html(cypherResult.rows.length)} rows</p>`);
        } else {
          $$renderer3.push("<!--[-1-->");
          $$renderer3.push(`<p class="text-muted-foreground text-sm">Run a Cypher query to see rows here.</p>`);
        }
        $$renderer3.push(`<!--]--></div>`);
      } else {
        $$renderer3.push("<!--[-1-->");
        $$renderer3.push(`<pre class="h-full overflow-auto p-3 text-xs">${escape_html(cypherResult ? JSON.stringify(cypherResult.rows, null, 2) : "Run a Cypher query to see JSON here.")}</pre>`);
      }
      $$renderer3.push(`<!--]--></div> `);
      if (cypherOpen || view !== "graph") {
        $$renderer3.push("<!--[0-->");
        $$renderer3.push(`<aside class="border-border bg-card/20 w-80 shrink-0 space-y-2 overflow-auto border-l p-3"><div><p class="text-foreground text-sm font-semibold">Example questions</p> <p class="text-muted-foreground text-xs">Click one to run it — results show on the left.</p></div> <!--[-->`);
        const each_array_6 = ensure_array_like(HELP_EXAMPLES);
        for (let $$index_6 = 0, $$length = each_array_6.length; $$index_6 < $$length; $$index_6++) {
          let ex = each_array_6[$$index_6];
          $$renderer3.push(`<button class="border-border bg-background hover:border-primary/50 hover:bg-secondary/50 block w-full space-y-1 rounded-md border p-2.5 text-left transition-colors"><span class="text-foreground block text-sm font-medium">${escape_html(ex.label)}</span> <span class="text-muted-foreground block text-xs leading-snug">${escape_html(ex.desc)}</span> <code class="text-muted-foreground/80 mt-1 block truncate rounded bg-muted/50 px-1.5 py-0.5 font-mono text-[10px]">${escape_html(ex.query)}</code></button>`);
        }
        $$renderer3.push(`<!--]--></aside>`);
      } else {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--></div></div> `);
      if (detail?.entity) {
        $$renderer3.push("<!--[0-->");
        const e = detail.entity;
        $$renderer3.push(`<aside class="border-border bg-card/30 w-80 shrink-0 overflow-auto border-l p-3"><div class="flex items-start justify-between gap-2"><h2 class="text-foreground text-sm font-semibold break-words">${escape_html(e.name)}</h2> <button class="text-muted-foreground hover:text-foreground text-xs">✕</button></div> <div class="text-muted-foreground mt-1 flex flex-wrap gap-1 text-xs"><span class="bg-secondary rounded px-1.5 py-0.5">${escape_html(e.entity_type)}</span> <span class="bg-secondary rounded px-1.5 py-0.5">${escape_html(e.mention_count)} mentions</span></div> `);
        if (detail.clips.length > 0) {
          $$renderer3.push("<!--[0-->");
          $$renderer3.push(`<h3 class="text-muted-foreground mt-4 mb-1 text-xs font-medium tracking-wide uppercase">Clips (${escape_html(detail.clips.length)})</h3> <ul class="flex flex-col gap-1"><!--[-->`);
          const each_array_7 = ensure_array_like(detail.clips);
          for (let $$index_7 = 0, $$length = each_array_7.length; $$index_7 < $$length; $$index_7++) {
            let c = each_array_7[$$index_7];
            $$renderer3.push(`<li><button class="hover:bg-secondary/60 w-full rounded p-1.5 text-left" title="Open in player"><div class="text-foreground flex items-center gap-1 text-xs"><span class="text-primary font-mono">${escape_html(fmtTime(c.start))}</span> <span class="text-muted-foreground truncate">${escape_html(c.title)}</span></div> `);
            if (c.text) {
              $$renderer3.push("<!--[0-->");
              $$renderer3.push(`<p class="text-muted-foreground mt-0.5 line-clamp-2 text-xs">${escape_html(c.text)}</p>`);
            } else {
              $$renderer3.push("<!--[-1-->");
            }
            $$renderer3.push(`<!--]--></button></li>`);
          }
          $$renderer3.push(`<!--]--></ul>`);
        } else {
          $$renderer3.push("<!--[-1-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (detail.neighbors.length > 0) {
          $$renderer3.push("<!--[0-->");
          $$renderer3.push(`<h3 class="text-muted-foreground mt-4 mb-1 text-xs font-medium tracking-wide uppercase">Related</h3> <div class="flex flex-wrap gap-1"><!--[-->`);
          const each_array_8 = ensure_array_like(detail.neighbors);
          for (let $$index_8 = 0, $$length = each_array_8.length; $$index_8 < $$length; $$index_8++) {
            let nb = each_array_8[$$index_8];
            $$renderer3.push(`<button class="bg-secondary hover:bg-secondary/70 rounded px-1.5 py-0.5 text-xs"${attr("title", nb.description || nb.name)}>${escape_html(nb.direction === "out" ? "→ " : "← ")}${escape_html(nb.name)}</button>`);
          }
          $$renderer3.push(`<!--]--></div>`);
        } else {
          $$renderer3.push("<!--[-1-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (detail.cooccur.length > 0) {
          $$renderer3.push("<!--[0-->");
          $$renderer3.push(`<h3 class="text-muted-foreground mt-4 mb-1 text-xs font-medium tracking-wide uppercase">Co-occurs with</h3> <div class="flex flex-wrap gap-1"><!--[-->`);
          const each_array_9 = ensure_array_like(detail.cooccur);
          for (let $$index_9 = 0, $$length = each_array_9.length; $$index_9 < $$length; $$index_9++) {
            let co = each_array_9[$$index_9];
            $$renderer3.push(`<button class="bg-secondary hover:bg-secondary/70 rounded px-1.5 py-0.5 text-xs">${escape_html(co.name)} <span class="text-muted-foreground">·${escape_html(co.shared)}</span></button>`);
          }
          $$renderer3.push(`<!--]--></div>`);
        } else {
          $$renderer3.push("<!--[-1-->");
        }
        $$renderer3.push(`<!--]--></aside>`);
      } else {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}
export {
  _page as default
};
