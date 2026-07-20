import "../../../chunks/async.js";
import { s as sanitize_props, a as spread_props, b as slot, i as attr_class, l as ensure_array_like, k as stringify, m as escape_html, d as derived } from "../../../chunks/renderer.js";
import { I as Icon } from "../../../chunks/Icon.js";
import { S as Search } from "../../../chunks/search.js";
import { M as Map$1 } from "../../../chunks/map.js";
import { A as Arrow_right } from "../../../chunks/arrow-right.js";
function Sparkles($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"
      }
    ],
    ["path", { "d": "M20 3v4" }],
    ["path", { "d": "M22 5h-4" }],
    ["path", { "d": "M4 17v2" }],
    ["path", { "d": "M5 18H3" }]
  ];
  Icon($$renderer, spread_props([
    { name: "sparkles" },
    $$sanitized_props,
    {
      /**
       * @component @name Sparkles
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNOS45MzcgMTUuNUEyIDIgMCAwIDAgOC41IDE0LjA2M2wtNi4xMzUtMS41ODJhLjUuNSAwIDAgMSAwLS45NjJMOC41IDkuOTM2QTIgMiAwIDAgMCA5LjkzNyA4LjVsMS41ODItNi4xMzVhLjUuNSAwIDAgMSAuOTYzIDBMMTQuMDYzIDguNUEyIDIgMCAwIDAgMTUuNSA5LjkzN2w2LjEzNSAxLjU4MWEuNS41IDAgMCAxIDAgLjk2NEwxNS41IDE0LjA2M2EyIDIgMCAwIDAtMS40MzcgMS40MzdsLTEuNTgyIDYuMTM1YS41LjUgMCAwIDEtLjk2MyAweiIgLz4KICA8cGF0aCBkPSJNMjAgM3Y0IiAvPgogIDxwYXRoIGQ9Ik0yMiA1aC00IiAvPgogIDxwYXRoIGQ9Ik00IDE3djIiIC8+CiAgPHBhdGggZD0iTTUgMThIMyIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/sparkles
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
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const JUDGES = [
      {
        icon: "⌨️",
        name: "Keyword",
        mode: "fts",
        accent: "border-l-sky-400",
        what: "exact words you type (FTS / BM25 on the transcript)",
        give: "your query words",
        vs: "transcript text (BM25). Good at exact terms, names."
      },
      {
        icon: "💬",
        name: "Vector",
        mode: "semantic",
        accent: "border-l-primary",
        what: "what a clip is about, even in other words (text vector)",
        give: "your text → a vector",
        vs: "text_embedding of each transcript. Good at topics, paraphrases."
      },
      {
        icon: "🖼️",
        name: "Image",
        mode: "visual",
        accent: "border-l-amber-400",
        what: "how much the video frame looks like your image (frame vector)",
        give: "your image → a vector",
        vs: "frame_embedding of each video frame. Good at visually similar scenes."
      },
      {
        icon: "🎬",
        name: "Scene",
        mode: "scene",
        accent: "border-l-emerald-400",
        what: "what's visible on screen, from each frame's Swedish caption (caption vector)",
        give: "your text → a vector",
        vs: 'caption_embedding of each frame’s Swedish caption. Good at what’s on screen ("plakat", "snöig gata").'
      }
    ];
    const LANES = [
      {
        name: "Keyword",
        sub: "transcript words",
        scan: "bg-sky-400",
        text: "text-sky-500 dark:text-sky-300"
      },
      {
        name: "Vector",
        sub: "transcript meaning",
        scan: "bg-primary",
        text: "text-primary"
      },
      {
        name: "Image",
        sub: "video frames",
        scan: "bg-amber-400",
        text: "text-amber-500 dark:text-amber-300"
      },
      {
        name: "Scene",
        sub: "frame captions",
        scan: "bg-emerald-400",
        text: "text-emerald-500 dark:text-emerald-300"
      }
    ];
    const MODES = [
      {
        mode: "Keyword",
        passes: "1 pass",
        detail: "BM25 word-match on the transcript",
        fuse: "—"
      },
      {
        mode: "Vector",
        passes: "1 pass",
        detail: "meaning-match on the transcript",
        fuse: "—"
      },
      {
        mode: "Scene",
        passes: "1 pass",
        detail: "matches the frame’s Swedish caption (meaning or keyword)",
        fuse: "—"
      },
      {
        mode: "Image",
        passes: "1 pass",
        detail: "image-similarity on the video frames (no text)",
        fuse: "—"
      },
      {
        mode: "Hybrid",
        passes: "2 passes",
        detail: "Keyword + Vector on the transcript",
        fuse: "RRF, or the Balance slider"
      },
      {
        mode: "Image + text",
        passes: "≤ 4 passes",
        detail: "Keyword + Vector + Image + Scene — the dial is ignored",
        fuse: "equal-weight RRF"
      }
    ];
    const SIGNAL_EXAMPLES = [
      {
        name: "Keyword",
        icon: "⌨️",
        accent: "border-l-sky-400",
        q: "Kristersson · phrase: vi inför ett exportstopp",
        finds: "every chunk whose transcript literally says it, ranked by BM25. Fat-fingered 'Kristerson'? Turn on Fuzzy and the typo still lands.",
        misses: "only the words actually spoken — 'uppsägningar' won't match a clip where the minister said 'varsel'. It never reads the frame.",
        use: "you know the exact word, name, or quote that was said."
      },
      {
        name: "Vector",
        icon: "💬",
        accent: "border-l-primary",
        q: "fler jobb",
        finds: "a minister saying 'vi måste öka sysselsättningen' — no shared word, same meaning (they sit close in embedding space).",
        misses: "can drift to merely-adjacent topics ('tillväxt'); reads only the transcript, so an unspoken on-screen chart is Scene's job.",
        use: "you remember the gist, not the exact words."
      },
      {
        name: "Scene",
        icon: "🎬",
        accent: "border-l-emerald-400",
        q: "snöig gata",
        finds: "an outdoor stakeout on a snow-covered street — even though the transcript only discusses energy policy. It matched the frame's Swedish caption, not the speech.",
        misses: "reads only the AI caption of what's on screen (as good as the captioner). Not face recognition — it finds the snow, not a person.",
        use: "you remember what the clip looked like, not what was said."
      },
      {
        name: "Image",
        icon: "🖼️",
        accent: "border-l-amber-400",
        q: "attach a photo of a podium with microphones",
        finds: "moments that LOOK the same — ministers at similar podiums, mic-cluster stakeouts. Swap in a snowy street → outdoor winter frames.",
        misses: "look-alike only, NOT face / identity. Add any text and the dial is ignored — your image becomes one equal vote among up to 4 passes, never a gate.",
        use: "you have a picture of a scene and want frames that look like it."
      }
    ];
    const K = 60;
    const keywordRanked = ["C", "A", "B"];
    const meaningRanked = ["A", "C", "D"];
    const rrfRows = (() => {
      const rankOf = (list) => new Map(list.map((c, i) => [c, i]));
      const kr = rankOf(keywordRanked);
      const mr = rankOf(meaningRanked);
      const clips = [.../* @__PURE__ */ new Set([...keywordRanked, ...meaningRanked])];
      return clips.map((clip) => {
        const parts = [];
        const k = kr.get(clip);
        if (k !== void 0) parts.push({ src: "⌨️", rank: k });
        const m = mr.get(clip);
        if (m !== void 0) parts.push({ src: "💬", rank: m });
        const score = parts.reduce((a, p) => a + 1 / (K + p.rank), 0);
        return { clip, parts, score, inBoth: parts.length === 2 };
      }).sort((a, b) => b.score - a.score);
    })();
    const SETTINGS = [
      {
        name: "Results to return",
        desc: "How many chunks to show (N, default 100)."
      },
      {
        name: "Rerank top",
        desc: "Head size K the cross-encoder re-reads (default 20)."
      },
      {
        name: "Balance",
        desc: "Keyword ↔ meaning weight. Hybrid only (ignored once an image is added)."
      },
      {
        name: "Match style",
        desc: "loose words, exact phrase, or fuzzy (typo-tolerant) keyword matching."
      }
    ];
    const LIMITS = [
      "Image search = visual frame similarity, not face / identity recognition.",
      "No speaker diarization — nothing links who is on screen to who is speaking.",
      "The reranker is text-only; it never uses the image.",
      "Multi-leg fusion (up to 4 judges) is equal-weight — there is no per-leg weight yet.",
      "Scene depends on AI captions — only as accurate as the frame captioner."
    ];
    const DECISION = [
      {
        when: "You can quote the exact words or a name that was said",
        then: "Keyword — type the phrase (Phrase for exact order, Fuzzy for typos). Turn on Rerank to float the closest wording up."
      },
      {
        when: "You only remember the gist, not the words",
        then: "Vector — describe the idea in your own words; it matches meaning, so paraphrases and synonyms still land."
      },
      {
        when: "You remember what it LOOKED like, not what was said",
        then: "Scene — describe the picture (podium, placard, snowy street). It searches the AI's caption of each frame, not the pixels or the speech."
      },
      {
        when: "You have a screenshot / photo from the clip",
        then: "Image — attach it to find visual look-alikes. Look-alike only, not face / identity; add words if you also recall what was said."
      },
      {
        when: "You want exact words AND meaning",
        then: "Hybrid — runs Keyword + Vector together; nudge the Balance slider toward keyword (precise) or meaning (looser)."
      },
      {
        when: "You're not sure which clue is strongest",
        then: "Attach an image and add text → all-angles mode: up to four legs (Keyword, Vector, Scene, Image) vote equally; the dial is ignored."
      }
    ];
    const TROUBLESHOOT = [
      {
        symptom: "A top hit has nothing to do with what is on screen",
        fix: "It won on the transcript — someone SAID your words even if the frame shows something else. Use Scene if you meant what was SHOWN."
      },
      {
        symptom: "You attached an image but look-alikes are not dominating",
        fix: "Expected: in mixed search the image is one equal vote (RRF), never a gate. Drop the text for Image-only, or add Scene to reinforce the look."
      },
      {
        symptom: "Fewer results came back than the count you set",
        fix: "A Filter is dropping them — a postfilter trims after searching. Loosen/remove it, switch it to prefilter, or use Vector to widen recall."
      },
      {
        symptom: "Off-topic hits on top, good ones lower down",
        fix: "Turn on Rerank to re-sort the top by transcript, push the Balance slider toward the signal you trust, or add a Filter."
      },
      {
        symptom: "Results feel noisy, scattered across press conferences",
        fix: "Narrow with Filters (name, language, referenskod). With prefilter on they narrow the transcript legs (Keyword/Vector) before searching; the Image/Scene legs always filter after ranking."
      },
      {
        symptom: "Sure a chunk exists but nothing finds it",
        fix: "Widen recall: add signals (Hybrid or Scene), raise the result count, clear Filters. Wrong exact words? Vector catches the meaning Keyword misses."
      },
      {
        symptom: "Rerank is on but the order barely changed",
        fix: "Rerank reads transcripts only — a no-op for Image, and it ignores captions. For visual / scene intent, fix the query or signal instead."
      }
    ];
    const GLOSSARY = [
      {
        term: "Chunk",
        def: "one flashcard — a few seconds of clip, with the spoken transcript on one side and the video frame (+ caption) on the other. The smallest thing search returns."
      },
      {
        term: "Caption",
        def: "a short Swedish sentence an AI wrote about what is VISIBLE in the frame. Lets you find things shown but never spoken."
      },
      {
        term: "Keyword (BM25)",
        def: "the literal-minded librarian — finds the exact words you typed (with phrases + small typo-tolerance) in the transcript. Great for names/quotes, blind to synonyms."
      },
      {
        term: "Embedding / Vector",
        def: "turns text into a dot on a giant 'meaning map' so similar ideas land near each other. Vector search returns transcripts whose dot sits closest to your query, even with no shared words."
      },
      {
        term: "Cosine",
        def: 'the ruler for "how close" two dots point on the meaning map. Same direction = alike. Just the yardstick, not a setting.'
      },
      {
        term: "RRF",
        def: "the fair vote-counter — rewards results that rank HIGH on MORE lists, using ranks not raw scores. (The Balance slider is the exception: it blends raw scores, Hybrid only.)"
      },
      {
        term: "Cross-encoder / Rerank",
        def: "the careful second reader — re-reads each candidate's transcript against your text and reshuffles the head. It scores by transcript only, so it can't use the image or caption signal — but it still re-ranks whatever candidates sit in the head."
      },
      {
        term: "Prefilter vs Postfilter",
        def: "Filters are a guest list. Prefilter checks IDs at the door (search only looks inside allowed chunks); Postfilter lets everyone search then removes the uninvited after — so you may get fewer results than you asked for."
      },
      {
        term: "Scene",
        def: "searching the PICTURE side with text — compares your words to the frame captions, surfacing what is on screen even when unspoken."
      },
      {
        term: "Image",
        def: "drag in a picture to find visual look-alikes (similar frames). Matches appearance, not who a person is."
      }
    ];
    const JOURNEY = [
      {
        label: "You ask",
        detail: "your text and/or attached image (plus any Filters) go to the backend."
      },
      {
        label: "Words & pictures → vectors",
        detail: "the embedding service (a vLLM model) turns your text into a meaning-vector and your image into an image-vector. Skipped for pure Keyword and Scene-keyword — those are word-matching, no vectors."
      },
      {
        label: "Each signal queries a table",
        detail: "Keyword & Vector read the chunks table (transcripts); Image & Scene read the chunk_frames table (one row per video frame), then join each frame back to its chunk."
      },
      {
        label: "Merge",
        detail: "the backend fuses the per-signal ranked lists — RRF, or Lance's native hybrid / the Balance slider."
      },
      {
        label: "Optional rerank",
        detail: "the reranker service (a second vLLM model) re-reads the top transcripts and reorders just the head."
      },
      {
        label: "Dress the results",
        detail: "the backend attaches each hit's caption and word-level alignments."
      },
      {
        label: "Back to you",
        detail: "one uniform result shape for every mode; click a hit and the player plays it with the transcript synced."
      }
    ];
    const SCENARIOS = [
      {
        give: "Keyword — text, no image",
        mode: "fts",
        passes: "Keyword (BM25)",
        combined: "single list",
        slider: "—",
        rerank: "yes"
      },
      {
        give: "Vector — text, no image",
        mode: "semantic",
        passes: "Vector (text embedding)",
        combined: "single list",
        slider: "—",
        rerank: "yes"
      },
      {
        give: "Scene — text, no image",
        mode: "scene",
        passes: "Scene caption (meaning, or BM25 via the Scene toggle)",
        combined: "single list",
        slider: "—",
        rerank: "yes"
      },
      {
        give: "Hybrid — text, no image",
        mode: "hybrid",
        passes: "Keyword + Vector",
        combined: "RRF, or the Balance slider",
        slider: "available",
        rerank: "yes"
      },
      {
        give: "Image only — no text",
        mode: "visual",
        passes: "Image (frame)",
        combined: "single list",
        slider: "—",
        rerank: "no-op (no text)"
      },
      {
        give: "Keyword + image + text",
        mode: "all",
        passes: "Keyword + Vector + Image + Scene",
        combined: "equal-weight RRF",
        slider: "ignored",
        rerank: "yes",
        highlight: true
      },
      {
        give: "Vector + image + text",
        mode: "all",
        passes: "Keyword + Vector + Image + Scene",
        combined: "equal-weight RRF",
        slider: "ignored",
        rerank: "yes",
        highlight: true
      },
      {
        give: "Scene + image + text",
        mode: "all",
        passes: "Keyword + Vector + Image + Scene",
        combined: "equal-weight RRF",
        slider: "ignored",
        rerank: "yes",
        highlight: true
      },
      {
        give: "Hybrid + image + text",
        mode: "all",
        passes: "Keyword + Vector + Image + Scene",
        combined: "equal-weight RRF",
        slider: "ignored",
        rerank: "yes",
        highlight: true
      }
    ];
    let dial = "hybrid";
    const demoMode = derived(() => {
      return "hybrid";
    });
    const demoLegs = derived(() => {
      switch (demoMode()) {
        case "all":
          return ["keyword", "vector", "image", "scene"];
        case "hybrid":
          return ["keyword", "vector"];
        case "fts":
          return ["keyword"];
        case "semantic":
          return ["vector"];
        case "scene":
          return ["scene"];
        case "visual":
          return ["image"];
        default:
          return [];
      }
    });
    const demoSlider = derived(() => demoMode() === "hybrid");
    const demoRerank = derived(() => demoMode() !== "visual" && demoMode() !== null);
    const DEMO_DIALS = ["keyword", "vector", "scene", "hybrid"];
    const DEMO_LANES = [
      {
        key: "keyword",
        name: "Keyword",
        icon: "⌨️",
        side: "said",
        on: "border-sky-400 bg-sky-400/10 text-sky-600 dark:text-sky-300"
      },
      {
        key: "vector",
        name: "Vector",
        icon: "💬",
        side: "said",
        on: "border-primary bg-primary/10 text-primary"
      },
      {
        key: "image",
        name: "Image",
        icon: "🖼️",
        side: "shown",
        on: "border-amber-400 bg-amber-400/10 text-amber-600 dark:text-amber-300"
      },
      {
        key: "scene",
        name: "Scene",
        icon: "🎬",
        side: "shown",
        on: "border-emerald-400 bg-emerald-400/10 text-emerald-600 dark:text-emerald-300"
      }
    ];
    let exIdx = 0;
    const WORKED = [
      {
        key: "hybrid",
        tab: "Paraphrase → Hybrid",
        title: "A minister talked about helping households with their electricity bills — but I forget the exact words.",
        mode: "hybrid",
        why: "You half-remember the topic but not the phrasing, so you need exact-word AND meaning matching at once — that is Hybrid (Keyword + Vector).",
        steps: [
          {
            stage: "Filters (none set)",
            yields: "No filter → both legs search the whole corpus, nothing narrowed away."
          },
          {
            stage: "Keyword leg — BM25 on transcript",
            yields: "Matches only literal spoken words of your query 'hjälpa hushåll med elräkningar'. #1 A 'hjälpa hushåll som inte klarar sina elräkningar'; #2 E 'hushållens elräkningar har tredubblats'. The paraphrase C is ABSENT — it never says your words, so BM25 scores it 0."
          },
          {
            stage: "Vector leg — cosine on text_embedding",
            yields: "Matches meaning, ignores shared words. #1 C 'vi vill stötta hushållen med elkostnaderna'; #2 A. C wins here despite zero word overlap with your query."
          },
          {
            stage: "RRF fuse — rank-based, equal weight",
            yields: "A is on BOTH lists (Keyword #1, Vector #2) → highest combined score. C is on ONE list but at #1 → just behind. Fused order: A, C, then the rest. A leads only because it scored on both legs."
          },
          {
            stage: "Rerank top-K — cross-encoder, transcript-only",
            yields: "Re-reads each candidate's transcript against your text. Recognises 'stötta hushållen med elkostnaderna' as the closest paraphrase and floats C to #1, A #2. (Ignores frames/captions.)"
          }
        ],
        result: "C — 'vi vill stötta hushållen med elkostnaderna' — wins, despite sharing no word with your query. Vector found it by meaning, RRF kept it near the top, and the text-only rerank promoted it to outright #1.",
        takeaway: "Remember the idea but not the words? Hybrid covers both — Keyword nails any literal word that was said, Vector rescues the differently-phrased clip, and rerank breaks the tie by re-reading transcripts."
      },
      {
        key: "scene",
        tab: "Snowy street → Scene",
        title: "It was filmed on a snowy street — but I forget what they said.",
        mode: "scene",
        why: "You remember the picture, not the words — so the only signal that can win is the one that reads each frame's caption of what is VISIBLE. That's Scene.",
        steps: [
          {
            stage: "Pick the mode",
            yields: "Type the picture in words: 'snöig gata utomhus', dial = Scene (no image attached, so not 'all'). Backend mode = scene."
          },
          {
            stage: "Why NOT Keyword",
            yields: "Keyword runs BM25 on the TRANSCRIPT only. This clip's transcript is energy talk ('elpriserna har stigit...') — 'snö' / 'gata' are never spoken, so BM25 scores it 0. It would instead surface a clip that literally says 'gata' (a street-repair debate) — the wrong scene."
          },
          {
            stage: "Why NOT Vector",
            yields: "Vector embeds the TRANSCRIPT and compares to your query. The transcript is about elpriser — far from 'snöig gata' in meaning space — so the clip ranks ~#80, effectively missed. Vector reads what was SAID, never what is shown."
          },
          {
            stage: "Scene leg — the one pass",
            yields: "Your text → one vector, compared by cosine to every frame's caption_embedding, then joined back to its chunk. Top captions: #1 .81 'minister intervjuas på en snöig gata utomhus'; #2 .77 'reporter på en snötäckt trottoar'; #3 .69 'vinterväg med bilar, snö faller'."
          },
          {
            stage: "Fuse — nothing to fuse",
            yields: "Scene is a single leg: no RRF, no Balance slider. The cosine-ranked caption list IS the result."
          },
          {
            stage: "Rerank — would it help? No",
            yields: "Rerank re-reads only each candidate's TRANSCRIPT vs your text. 'snöig gata' has no match in the energy transcript, so it would score the right clip LOW and could push it down — and it never reads the caption that won. Leave it off for visual intent."
          }
        ],
        result: "The chunk behind caption #1 — 'minister intervjuas på en snöig gata utomhus' (.81). Scene matched your text against the frame's caption of what's on screen, not the spoken energy transcript that Keyword and Vector were stuck reading.",
        takeaway: "Remember the LOOK, not the words? Use Scene — it reads each frame's caption, so a clip whose transcript is pure energy policy still surfaces from its snowy-street visual. Don't lean on Rerank, which only re-reads transcripts."
      },
      {
        key: "all",
        tab: "Photo + word → all",
        title: "Here's a screenshot of a podium with mics — and I think it was the press conference about the export stop.",
        mode: "all",
        why: "Two independent clues — a photo AND a remembered word ('exportstopp'). Attach the image and type the text → the 4-leg 'all' mode, where each clue votes equally.",
        steps: [
          {
            stage: "Build the spec — dial IGNORED",
            yields: "You drag in podium.png and type 'exportstopp'. Image + text → mode = all no matter where the dial sits; the Balance slider is dropped. Your text drives the Keyword + Vector + Scene legs; the image drives the Image leg. Each leg over-fetches ~3×N candidates."
          },
          {
            stage: "Leg 1 — Keyword (BM25 on transcript)",
            yields: "Literal 'exportstopp': #1 chunk_842 'vi inför ett exportstopp för...'; #2 chunk_119 'exportstoppet träder i kraft'. Even though you 'meant' Vector by habit, your text still ran a real BM25 keyword leg."
          },
          {
            stage: "Leg 2 — Vector (meaning on text_embedding)",
            yields: "Paraphrases with no shared word: #1 chunk_507 'stoppa utförseln av krigsmateriel'; #2 chunk_842. chunk_842 ranks well here too."
          },
          {
            stage: "Leg 3 — Image (frame_embedding vs your photo)",
            yields: "Pure look-alike, no text: #1 chunk_077 (near-identical mic cluster); #2 chunk_842 (same podium room); #3 chunk_415. It found visually-similar podiums; it does NOT know who is standing there."
          },
          {
            stage: "Leg 4 — Scene (caption_embedding vs your text)",
            yields: "Your text vs each frame's caption: #1 chunk_842 'minister vid talarstol, plakat om vapenexport'; #2 chunk_203 'diagram över exportsiffror'. Your text drove a second picture-side leg."
          },
          {
            stage: "RRF fuse — equal weight, rank-based",
            yields: "chunk_842 appears near the top of ALL FOUR lists → highest combined score. chunk_077 is in 2 lists, chunk_507 in 2. Nobody is the unanimous #1, but chunk_842 is high on the MOST lists → it wins. The image was one vote of four, never a gate."
          },
          {
            stage: "Rerank the head — text-only, top-K",
            yields: "Reads ONLY transcripts vs 'exportstopp'. chunk_842 literally says 'vi inför ett exportstopp' → stays #1. A look-alike like chunk_415 (right podium, transcript about elpriser) gets pushed DOWN — rerank can't see its frame, only its off-topic words."
          }
        ],
        result: "chunk_842 — the minister at the podium actually announcing the export stop. It was the only chunk near the top of all four legs at once; RRF rewards that agreement, and the text-only rerank confirmed it via its on-topic transcript.",
        takeaway: "Two kinds of memory — a picture AND a word? Attach both: 'all' runs four equal legs (the dial is irrelevant), so your text still fires a BM25 leg and a Scene leg alongside the image. The clip high on the most legs wins; the image is a vote, never a filter."
      },
      {
        key: "postfilter",
        tab: "Too few → prefilter",
        title: "I filtered by a minister's name and got far fewer hits than my count — and the good ones vanished.",
        mode: "hybrid",
        why: "You want every clip where Andersson discusses elstöd, so Hybrid gives the best recall — but the Filter is set to postfilter, which silently drops most of each leg's results.",
        steps: [
          {
            stage: "Setup",
            yields: "Mode = Hybrid. Query 'elstöd till hushållen'. Count N = 20. Filter namn LIKE '%Andersson%'. Filter toggle = POSTFILTER. Hybrid runs Keyword + Vector over the whole corpus, then fuses by RRF."
          },
          {
            stage: "Postfilter = rank everyone, then drop",
            yields: "Each leg ranks the WHOLE corpus (mostly other ministers) and keeps its candidate pool, THEN deletes the non-Andersson rows. Andersson is only a fraction of each pool, so few survive — the fused list comes back well short of 20."
          },
          {
            stage: "The vanished clip",
            yields: "A clip you remember ('elstödet kommer redan i januari', Andersson) ranks low corpus-wide — it competes against every minister — so it falls outside each leg's fetched pool and is dropped before you ever see it. That's your missing good one."
          },
          {
            stage: "Diagnosis",
            yields: "Postfilter applies the Filter AFTER each leg ranks the full corpus, so your count (20) is the size BEFORE the drop, not what is delivered — and anything ranked low across the whole corpus is invisible."
          },
          {
            stage: "Fix — flip to PREFILTER",
            yields: "Set the Filter to prefilter (the default; the backend passes prefilter=True). Now the Filter narrows the corpus to ONLY Andersson's chunks BEFORE either leg ranks."
          },
          {
            stage: "Result after prefilter",
            yields: "Each leg now ranks within Andersson only, so it returns a full 20 — and 'elstödet kommer redan i januari' competes only against other Andersson clips, where it ranks near the top of both legs, so RRF puts it high."
          }
        ],
        result: "Flipping postfilter → prefilter turns ~5 results into a full 20, and the previously-missing clip lands near the top — because prefilter lets it compete only against Andersson's chunks instead of the whole corpus.",
        takeaway: "A metadata Filter on postfilter ranks the whole corpus then DROPS non-matches → fewer than your count, and any match ranked low corpus-wide disappears. Switch to prefilter to narrow first and let every leg return a full N from that subset."
      }
    ];
    $$renderer2.push(`<div class="h-full overflow-y-auto svelte-36n0qb"><div class="mx-auto max-w-3xl px-6 py-8 svelte-36n0qb"><div class="mb-6 svelte-36n0qb"><div class="text-primary mb-1 flex items-center gap-2 text-xs font-medium tracking-wide uppercase svelte-36n0qb">`);
    Sparkles($$renderer2, { class: "size-3.5" });
    $$renderer2.push(`<!----> Guide</div> <h1 class="text-foreground text-2xl font-semibold tracking-tight svelte-36n0qb">How the viewer works</h1> <div class="border-border mt-4 flex gap-1 border-b svelte-36n0qb"><button type="button"${attr_class(
      "flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors border-primary text-foreground",
      "svelte-36n0qb"
    )}>`);
    Search($$renderer2, { class: "size-4" });
    $$renderer2.push(`<!----> Search</button> <button type="button"${attr_class(
      "flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors text-muted-foreground hover:text-foreground border-transparent",
      "svelte-36n0qb"
    )}>`);
    Map$1($$renderer2, { class: "size-4" });
    $$renderer2.push(`<!----> Atlas</button></div></div> `);
    {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="reveal bg-card border-border mb-6 rounded-lg border px-4 py-3 text-sm svelte-36n0qb"><span class="text-foreground svelte-36n0qb">Every result is one</span> <strong class="text-primary svelte-36n0qb">chunk</strong> <span class="text-muted-foreground svelte-36n0qb">= a short transcript span + the video frame from that moment. Search ranks chunks.</span></div> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Two worlds: was it said, or was it shown?</h2> <p class="text-muted-foreground mb-4 text-sm leading-relaxed svelte-36n0qb">The one question to ask before searching. Every chunk has two sides — the words that were <strong class="text-foreground svelte-36n0qb">said</strong> (the transcript) and the moment that was <strong class="text-foreground svelte-36n0qb">shown</strong> (the frame + its Swedish caption). Each signal
          reads one side, so figure out which side your memory lives on and the right tool almost picks
          itself.</p> <div class="grid gap-3 sm:grid-cols-2 svelte-36n0qb"><div class="reveal rounded-lg border border-sky-400/40 bg-sky-400/5 p-3 svelte-36n0qb"><div class="text-foreground font-medium svelte-36n0qb">🗣️ Said — the transcript</div> <p class="text-muted-foreground mt-1 text-xs leading-relaxed svelte-36n0qb">What someone spoke. Read by <span class="font-medium text-sky-500 dark:text-sky-300 svelte-36n0qb">Keyword</span> (exact words) and <span class="text-primary font-medium svelte-36n0qb">Vector</span> (the meaning, even
              in other words). Neither looks at the picture.</p></div> <div class="reveal rounded-lg border border-emerald-400/40 bg-emerald-400/5 p-3 svelte-36n0qb"><div class="text-foreground font-medium svelte-36n0qb">👁️ Shown — the frame</div> <p class="text-muted-foreground mt-1 text-xs leading-relaxed svelte-36n0qb">What was on screen. Read by <span class="font-medium text-emerald-500 dark:text-emerald-300 svelte-36n0qb">Scene</span> (describe it in words → matches the caption) and <span class="font-medium text-amber-500 dark:text-amber-300 svelte-36n0qb">Image</span> (hand it a photo → visual look-alikes).</p></div></div> <p class="text-muted-foreground mt-3 text-sm leading-relaxed svelte-36n0qb">Not sure which world? Attach an image and add words — that opens the all-angles mode,
          which votes across both worlds at once (the image is one vote, never the boss). And note: <strong class="text-foreground svelte-36n0qb">Rerank</strong> only re-reads the <em class="svelte-36n0qb">said</em> side, and <strong class="text-foreground svelte-36n0qb">Filters</strong> aren't a world — they narrow the room <em class="svelte-36n0qb">before</em> (or, as a postfilter, after) every search.</p></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">What "meaning search" actually is</h2> <p class="text-muted-foreground mb-3 text-sm leading-relaxed svelte-36n0qb">Picture a vast map where every idea gets a pin, and similar meanings land near each other
          — "the minister stepped down" sits a short walk from "the minister resigned", even with no
          shared word. An <strong class="text-foreground svelte-36n0qb">embedding</strong> is just that pin. Vector search drops your
          query on the same map and hands back the nearest chunks — so wording barely matters.</p> <div class="border-border bg-surface2 rounded-lg border p-3 text-xs leading-relaxed svelte-36n0qb"><ul class="text-muted-foreground space-y-1.5 svelte-36n0qb"><li class="svelte-36n0qb"><strong class="text-foreground svelte-36n0qb">Embedding = a map pin for meaning.</strong> Close = similar;
              far = unrelated. That distance is the search.</li> <li class="svelte-36n0qb"><strong class="text-foreground svelte-36n0qb">"Cosine" = pointing the same direction.</strong> Two ideas
              headed the same way are a match — length / loudness doesn't matter.</li> <li class="svelte-36n0qb"><strong class="text-foreground svelte-36n0qb">One shared map for text AND captions.</strong> Transcript-meanings
              and caption-meanings live in the same space, so typed words can find a frame's caption —
              that's the trick behind Scene.</li> <li class="svelte-36n0qb"><strong class="text-foreground svelte-36n0qb">Meaning, not letters.</strong> Vector shrugs off spelling
              but can drift on names / codes / quotes — that's where Keyword (or Filters) earns its keep.</li></ul></div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Why cast several nets, then pull them together</h2> <p class="text-muted-foreground mb-3 text-sm leading-relaxed svelte-36n0qb">No single signal sees the whole moment — Keyword and Vector only hear the words; Scene and
          Image only see the picture. So the app asks several witnesses at once — a stenographer, a
          paraphraser, a set photographer — and trusts the clips they <em class="svelte-36n0qb">agree</em> on. Agreement across
          independent signals is the strongest hint a clip is really relevant.</p> <div class="border-border bg-surface2 rounded-lg border p-3 text-xs leading-relaxed svelte-36n0qb"><ul class="text-muted-foreground space-y-1.5 svelte-36n0qb"><li class="svelte-36n0qb"><strong class="text-foreground svelte-36n0qb">Wide net first, tighten later.</strong> In the fused <code class="font-mono svelte-36n0qb">all</code> mode each leg over-fetches ~3× more candidates than you
              asked for (recall — don't miss it), then fusion + Rerank squeeze to the best few (precision).</li> <li class="svelte-36n0qb"><strong class="text-foreground svelte-36n0qb">Agreement is the relevance signal.</strong> Appearing once
              is a maybe; ranking high on three lists is a strong yes.</li> <li class="svelte-36n0qb"><strong class="text-foreground svelte-36n0qb">Votes, not vetoes.</strong> In image-plus-text search every
              leg is one equal vote (RRF); a missing leg just abstains and the rest decide.</li> <li class="svelte-36n0qb"><strong class="text-foreground svelte-36n0qb">Catches synonyms AND silence.</strong> Vector saves you
              when the words differ; Scene saves you when the thing on screen was never spoken.</li></ul></div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Is "image + keyword/vector" multiple passes, or a prefilter?</h2> <p class="text-muted-foreground mb-4 text-sm leading-relaxed svelte-36n0qb"><strong class="text-foreground svelte-36n0qb">Multiple independent passes, merged by rank — never a prefilter.</strong> The moment you attach an image <em class="svelte-36n0qb">and</em> have text, the Keyword/Vector/Hybrid dial stops
          mattering: the tool runs up to four searches side by side, each scanning the whole library on
          its own, then merges their rankings.</p> <div class="bg-card border-border rounded-xl border p-4 svelte-36n0qb"><div class="text-muted-foreground mb-2 text-center text-[11px] svelte-36n0qb">every pass starts from the <strong class="text-foreground svelte-36n0qb">same full library</strong> — all
            at once</div> <div class="grid grid-cols-2 gap-2 sm:grid-cols-4 svelte-36n0qb"><!--[-->`);
      const each_array = ensure_array_like(LANES);
      for (let i = 0, $$length = each_array.length; i < $$length; i++) {
        let lane = each_array[i];
        $$renderer2.push(`<div class="reveal bg-surface2 border-border relative overflow-hidden rounded-lg border p-2.5 text-center svelte-36n0qb"><div${attr_class(`${stringify(lane.text)} text-xs font-semibold`, "svelte-36n0qb")}>${escape_html(lane.name)}</div> <div class="text-muted-foreground text-[10px] svelte-36n0qb">${escape_html(lane.sub)}</div> <div class="text-muted-foreground mt-1 text-[10px] svelte-36n0qb">→ own ranked list</div> <span${attr_class(`lane-scan ${stringify(lane.scan)}`, "svelte-36n0qb")}></span></div>`);
      }
      $$renderer2.push(`<!--]--></div> <div class="text-muted-foreground my-1 text-center text-sm svelte-36n0qb">▼ ▼ ▼ ▼</div> <div class="border-primary/50 bg-primary/10 text-foreground rounded-lg border px-3 py-2 text-center text-sm font-medium svelte-36n0qb">Merge by rank — RRF (equal weight). High in <em class="svelte-36n0qb">more</em> lists → wins.</div> <div class="text-muted-foreground my-1 text-center text-sm svelte-36n0qb">▼</div> <div class="bg-background border-border text-foreground rounded-lg border px-3 py-2 text-center text-sm font-semibold svelte-36n0qb">Top N results</div></div> <div class="mt-3 grid gap-3 sm:grid-cols-2 svelte-36n0qb"><div class="reveal rounded-lg border border-emerald-500/40 bg-emerald-500/5 p-3 text-xs leading-relaxed svelte-36n0qb"><div class="mb-1 font-semibold text-emerald-600 dark:text-emerald-300 svelte-36n0qb">✓ What actually happens</div> <p class="text-muted-foreground svelte-36n0qb">Four searchlights sweep the same full library at once; their rankings are pooled. A
              chunk the image never matched can still rank #1 on the strength of its transcript
              alone — the image is a <strong class="text-foreground svelte-36n0qb">voice in the vote</strong>, not a gate.</p></div> <div class="reveal rounded-lg border border-red-500/40 bg-red-500/5 p-3 text-xs leading-relaxed svelte-36n0qb"><div class="mb-1 font-semibold text-red-600 line-through dark:text-red-300 svelte-36n0qb">✗ Not this</div> <p class="text-muted-foreground svelte-36n0qb">The image runs first, grabs a shortlist of look-alike clips, and the text then
              searches <em class="svelte-36n0qb">only inside</em> that shortlist. <strong class="text-foreground svelte-36n0qb">No such funnel exists.</strong> (Only the <em class="svelte-36n0qb">Filters</em> panel narrows first — see below.)</p></div></div> <div class="reveal bg-surface2 border-border mt-3 rounded-lg border p-3 text-xs leading-relaxed svelte-36n0qb"><div class="text-foreground mb-1 font-medium svelte-36n0qb">Traced: <code class="font-mono svelte-36n0qb">"arbete"</code> + a photo of a snowy street</div> <p class="text-muted-foreground svelte-36n0qb">Four passes fire at once. <strong class="text-foreground svelte-36n0qb">Keyword</strong> ranks a
            labour-market briefing #1; <strong class="text-foreground svelte-36n0qb">Vector</strong> surfaces a
            clip about "sysselsättning" (no exact word) at #2; <strong class="text-foreground svelte-36n0qb">Image</strong> ranks an outdoor winter stakeout by frame-similarity; <strong class="text-foreground svelte-36n0qb">Scene</strong> lifts a frame captioned "person på snöig gata".
            A clip that lands #3 / #5 / #3 / #2 across all four lists beats the briefing that was #1 in
            only one — agreement wins. The photo never walled off a subset that "arbete" then searched
            within.</p></div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">In what order does it happen? (and what's prioritised)</h2> <p class="text-muted-foreground mb-4 text-sm leading-relaxed svelte-36n0qb">The search legs have <strong class="text-foreground svelte-36n0qb">no priority over each other</strong> —
          they're equal-weight and run at the same time. So "image + Hybrid (FTS + Vector)" doesn't rank
          one before another; the only real ordering is these pipeline stages:</p> <ol class="space-y-3 svelte-36n0qb"><li class="reveal flex gap-3 svelte-36n0qb"><span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold svelte-36n0qb">1</span> <div class="svelte-36n0qb"><div class="text-foreground text-sm font-medium svelte-36n0qb">Filters first <span class="text-muted-foreground font-normal svelte-36n0qb">(optional)</span></div> <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed svelte-36n0qb">Your metadata Filters (language, name, referenskod, extraid, raw SQL) become a <code class="font-mono svelte-36n0qb">WHERE</code>. For the <strong class="text-foreground svelte-36n0qb">transcript legs</strong> (Keyword, Vector, Hybrid), <em class="svelte-36n0qb">prefilter</em> narrows the corpus <em class="svelte-36n0qb">before</em> they search (toggle off → after); the <strong class="text-foreground svelte-36n0qb">frame legs</strong> (Image, Scene) always apply it <em class="svelte-36n0qb">after</em> ranking. Either way it's the only narrowing step. No filters set → the
                whole library.</p></div></li> <li class="reveal flex gap-3 svelte-36n0qb"><span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold svelte-36n0qb">2</span> <div class="svelte-36n0qb"><div class="text-foreground text-sm font-medium svelte-36n0qb">Every leg runs in parallel</div> <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed svelte-36n0qb">Keyword, Vector, Image and Scene each search independently and <strong class="text-foreground svelte-36n0qb">at the same time</strong>. No leg goes first and none outranks another. In <code class="font-mono svelte-36n0qb">all</code> mode each over-fetches ~3×N candidates so good chunks aren't lost before the merge (single-leg
                searches fetch exactly your count).</p></div></li> <li class="reveal flex gap-3 svelte-36n0qb"><span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold svelte-36n0qb">3</span> <div class="svelte-36n0qb"><div class="text-foreground text-sm font-medium svelte-36n0qb">Fuse by rank — equal weight (RRF)</div> <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed svelte-36n0qb">Each leg's ranked list contributes <code class="font-mono svelte-36n0qb">1/(60 + rank)</code> per
                chunk; a chunk high in <em class="svelte-36n0qb">more</em> lists wins. Every leg counts the same. (Hybrid <em class="svelte-36n0qb">without</em> an image may instead use the Balance slider — a 2-way blend of raw scores;
                adding an image always reverts to equal-weight RRF.)</p></div></li> <li class="reveal flex gap-3 svelte-36n0qb"><span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold svelte-36n0qb">4</span> <div class="svelte-36n0qb"><div class="text-foreground text-sm font-medium svelte-36n0qb">Rerank the head <span class="text-muted-foreground font-normal svelte-36n0qb">(optional)</span></div> <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed svelte-36n0qb">A text-only cross-encoder re-scores just the top <em class="svelte-36n0qb">K</em> of the fused list,
                reading each chunk's transcript (the <code class="font-mono svelte-36n0qb">text</code> column) against
                your text — see below.</p></div></li> <li class="reveal flex gap-3 svelte-36n0qb"><span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold svelte-36n0qb">5</span> <div class="svelte-36n0qb"><div class="text-foreground text-sm font-medium svelte-36n0qb">Trim to top N</div> <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed svelte-36n0qb">The fused (and optionally reranked) list is cut to N — the results you see.</p></div></li></ol></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">The 4 judges</h2> <p class="text-muted-foreground mb-4 text-sm leading-relaxed svelte-36n0qb">A search asks up to four independent judges to rank the chunks. Each runs as its <em class="svelte-36n0qb">own pass</em> over its <em class="svelte-36n0qb">own</em> space and hands back its <em class="svelte-36n0qb">own</em> ranked list —
          they never compare scores with each other.</p> <div class="grid gap-3 sm:grid-cols-2 svelte-36n0qb"><!--[-->`);
      const each_array_1 = ensure_array_like(JUDGES);
      for (let i = 0, $$length = each_array_1.length; i < $$length; i++) {
        let j = each_array_1[i];
        $$renderer2.push(`<div${attr_class(`reveal bg-surface2 border-border ${stringify(j.accent)} rounded-lg border border-l-[3px] p-3`, "svelte-36n0qb")}><div class="flex items-center gap-2 svelte-36n0qb"><span class="text-base leading-none svelte-36n0qb">${escape_html(j.icon)}</span> <span class="text-foreground font-medium svelte-36n0qb">${escape_html(j.name)}</span> <code class="bg-card text-primary ml-auto rounded px-1.5 py-0.5 font-mono text-[11px] svelte-36n0qb">mode: ${escape_html(j.mode)}</code></div> <p class="text-muted-foreground mt-1.5 text-xs leading-relaxed svelte-36n0qb">${escape_html(j.what)}</p></div>`);
      }
      $$renderer2.push(`<!--]--></div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-3 text-lg font-semibold svelte-36n0qb">What each judge actually does</h2> <div class="border-border overflow-hidden rounded-lg border text-xs svelte-36n0qb"><div class="bg-surface2 text-foreground grid grid-cols-[84px_1fr_1.4fr] font-medium svelte-36n0qb"><div class="border-border border-b p-2 svelte-36n0qb">Judge</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">You give it</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">Compared against</div></div> <!--[-->`);
      const each_array_2 = ensure_array_like(JUDGES);
      for (let i = 0, $$length = each_array_2.length; i < $$length; i++) {
        let j = each_array_2[i];
        $$renderer2.push(`<div class="reveal text-muted-foreground grid grid-cols-[84px_1fr_1.4fr] last:[&amp;>div]:border-b-0 svelte-36n0qb"><div class="border-border text-foreground border-b p-2 svelte-36n0qb">${escape_html(j.icon)} ${escape_html(j.name)}</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">${escape_html(j.give)}</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">${escape_html(j.vs)}</div></div>`);
      }
      $$renderer2.push(`<!--]--></div> <div class="reveal border-primary/40 bg-primary/5 text-muted-foreground mt-3 rounded-lg border border-dashed p-3 text-sm svelte-36n0qb">💡 Your <strong class="text-foreground svelte-36n0qb">text</strong> drives <strong class="text-foreground svelte-36n0qb">two</strong> passes at once: the <span class="text-primary font-medium svelte-36n0qb">Meaning</span> leg (what was <em class="svelte-36n0qb">said</em>) and the <span class="font-medium text-emerald-500 dark:text-emerald-300 svelte-36n0qb">Scene</span> leg (what's <em class="svelte-36n0qb">shown</em>, from the frame caption).</div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">See each signal in action</h2> <p class="text-muted-foreground mb-4 text-sm leading-relaxed svelte-36n0qb">One concrete query for each — what it finds, what it misses, and when to reach for it.</p> <div class="grid gap-3 sm:grid-cols-2 svelte-36n0qb"><!--[-->`);
      const each_array_3 = ensure_array_like(SIGNAL_EXAMPLES);
      for (let i = 0, $$length = each_array_3.length; i < $$length; i++) {
        let s = each_array_3[i];
        $$renderer2.push(`<div${attr_class(`reveal bg-surface2 border-border ${stringify(s.accent)} rounded-lg border border-l-[3px] p-3 text-xs leading-relaxed`, "svelte-36n0qb")}><div class="mb-1.5 flex items-center gap-2 svelte-36n0qb"><span class="text-base leading-none svelte-36n0qb">${escape_html(s.icon)}</span> <span class="text-foreground font-medium svelte-36n0qb">${escape_html(s.name)}</span></div> <div class="mb-2 svelte-36n0qb"><code class="bg-card text-primary rounded px-1.5 py-0.5 font-mono text-[11px] svelte-36n0qb">${escape_html(s.q)}</code></div> <p class="text-muted-foreground svelte-36n0qb"><span class="font-medium text-emerald-600 dark:text-emerald-300 svelte-36n0qb">Finds:</span> ${escape_html(s.finds)}</p> <p class="text-muted-foreground mt-1 svelte-36n0qb"><span class="font-medium text-amber-600 dark:text-amber-300 svelte-36n0qb">Misses:</span> ${escape_html(s.misses)}</p> <p class="text-muted-foreground mt-1 svelte-36n0qb"><span class="text-foreground font-medium svelte-36n0qb">Use when:</span> ${escape_html(s.use)}</p></div>`);
      }
      $$renderer2.push(`<!--]--></div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Which search should I reach for?</h2> <p class="text-muted-foreground mb-4 text-sm leading-relaxed svelte-36n0qb">Pick by what your memory actually held onto — the exact words, the gist, or the look.</p> <div class="border-border overflow-hidden rounded-lg border text-xs svelte-36n0qb"><!--[-->`);
      const each_array_4 = ensure_array_like(DECISION);
      for (let i = 0, $$length = each_array_4.length; i < $$length; i++) {
        let r = each_array_4[i];
        $$renderer2.push(`<div class="reveal grid grid-cols-[1fr_1.4fr] last:[&amp;>div]:border-b-0 svelte-36n0qb"><div class="border-border text-foreground border-b p-2 font-medium svelte-36n0qb">${escape_html(r.when)}</div> <div class="border-border text-muted-foreground border-b border-l p-2 svelte-36n0qb">${escape_html(r.then)}</div></div>`);
      }
      $$renderer2.push(`<!--]--></div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-3 text-lg font-semibold svelte-36n0qb">How many passes each mode runs</h2> <div class="border-border overflow-hidden rounded-lg border text-xs svelte-36n0qb"><div class="bg-surface2 text-foreground grid grid-cols-[1fr_auto_1.6fr] font-medium svelte-36n0qb"><div class="border-border border-b p-2 svelte-36n0qb">Mode</div> <div class="border-border border-b border-l p-2 text-center svelte-36n0qb">Passes</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">What runs · fused by</div></div> <!--[-->`);
      const each_array_5 = ensure_array_like(MODES);
      for (let i = 0, $$length = each_array_5.length; i < $$length; i++) {
        let m = each_array_5[i];
        $$renderer2.push(`<div${attr_class(`reveal text-muted-foreground grid grid-cols-[1fr_auto_1.6fr] last:[&>div]:border-b-0 ${stringify(m.mode === "Image + text" ? "bg-highlight/10" : "")}`, "svelte-36n0qb")}><div class="border-border text-foreground border-b p-2 font-medium svelte-36n0qb">${escape_html(m.mode)}</div> <div class="border-border border-b border-l p-2 text-center font-mono svelte-36n0qb">${escape_html(m.passes)}</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">${escape_html(m.detail)}`);
        if (m.fuse !== "—") {
          $$renderer2.push("<!--[0-->");
          $$renderer2.push(` · <span class="text-foreground svelte-36n0qb">${escape_html(m.fuse)}</span>`);
        } else {
          $$renderer2.push("<!--[-1-->");
        }
        $$renderer2.push(`<!--]--></div></div>`);
      }
      $$renderer2.push(`<!--]--></div> <div class="reveal border-amber-500/40 bg-amber-500/5 text-muted-foreground mt-3 rounded-lg border p-3 text-sm leading-relaxed svelte-36n0qb">⚠️ <strong class="text-foreground svelte-36n0qb">Attaching an image overrides the dial.</strong> Image +
          text always becomes the 4-way <code class="font-mono svelte-36n0qb">all</code> mode regardless of whether
          you picked Keyword, Vector, or Hybrid. Image alone stays single-pass Image; text alone obeys
          the dial.</div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Scenario matrix: exactly what runs, and when</h2> <p class="text-muted-foreground mb-3 text-sm leading-relaxed svelte-36n0qb">Two levers decide everything. <strong class="text-foreground svelte-36n0qb">(1)</strong> the dial
          (Keyword / Vector / Scene / Hybrid) only matters <em class="svelte-36n0qb">when there's no image</em>. <strong class="text-foreground svelte-36n0qb">(2)</strong> attaching an image is the master switch —
          image + any text becomes the 4-way <code class="font-mono svelte-36n0qb">all</code> (dial &amp; slider
          ignored); image alone becomes <code class="font-mono svelte-36n0qb">visual</code>. Inside any multi-pass mode <strong class="text-foreground svelte-36n0qb">no leg is prioritised</strong> — fusion is equal-weight RRF
          (Hybrid may instead use the Balance slider).</p> <div class="bg-card border-border mb-3 rounded-xl border p-4 svelte-36n0qb"><div class="flex flex-wrap items-center gap-1.5 svelte-36n0qb"><span class="text-muted-foreground mr-1 text-xs svelte-36n0qb">Dial</span> <!--[-->`);
      const each_array_6 = ensure_array_like(DEMO_DIALS);
      for (let $$index_6 = 0, $$length = each_array_6.length; $$index_6 < $$length; $$index_6++) {
        let d = each_array_6[$$index_6];
        $$renderer2.push(`<button type="button"${attr_class(
          `rounded-md border px-2 py-1 text-xs font-medium capitalize transition-all ${stringify(dial === d ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground hover:text-foreground")} ${stringify("")}`,
          "svelte-36n0qb"
        )}>${escape_html(d)}</button>`);
      }
      $$renderer2.push(`<!--]--> <span class="bg-border mx-1.5 h-4 w-px svelte-36n0qb"></span> <button type="button"${attr_class(
        `rounded-md border px-2 py-1 text-xs font-medium transition-colors ${stringify(
          "border-foreground/40 text-foreground"
        )}`,
        "svelte-36n0qb"
      )}>${escape_html("✓ ")}Text</button> <button type="button"${attr_class(
        `rounded-md border px-2 py-1 text-xs font-medium transition-colors ${stringify("border-border text-muted-foreground")}`,
        "svelte-36n0qb"
      )}>${escape_html("")}🖼️ Image</button></div> <div class="mt-3 flex flex-wrap items-center gap-2 text-xs svelte-36n0qb"><span class="text-muted-foreground svelte-36n0qb">→ runs</span> `);
      if (demoMode()) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<code class="bg-primary/15 text-primary rounded px-2 py-0.5 font-mono font-semibold svelte-36n0qb">${escape_html(demoMode())}</code> <span class="text-muted-foreground svelte-36n0qb">${escape_html(demoLegs().length)} pass${escape_html(demoLegs().length > 1 ? "es" : "")}</span>`);
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`<span class="text-muted-foreground italic svelte-36n0qb">nothing — type text or attach an image</span>`);
      }
      $$renderer2.push(`<!--]--></div> <div class="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4 svelte-36n0qb"><!--[-->`);
      const each_array_7 = ensure_array_like(DEMO_LANES);
      for (let $$index_7 = 0, $$length = each_array_7.length; $$index_7 < $$length; $$index_7++) {
        let lane = each_array_7[$$index_7];
        const active = demoLegs().includes(lane.key);
        $$renderer2.push(`<div${attr_class(
          `rounded-lg border p-2.5 text-center transition-all duration-300 ${stringify(active ? lane.on : "border-border text-muted-foreground scale-95 opacity-40 grayscale")}`,
          "svelte-36n0qb"
        )}><div class="text-base leading-none svelte-36n0qb">${escape_html(lane.icon)}</div> <div class="mt-1 text-xs font-medium svelte-36n0qb">${escape_html(lane.name)}</div> <div class="text-[10px] opacity-70 svelte-36n0qb">${escape_html(lane.side)}</div></div>`);
      }
      $$renderer2.push(`<!--]--></div> <div class="mt-3 flex flex-wrap items-center gap-1.5 text-[11px] svelte-36n0qb"><span${attr_class(
        `rounded px-2 py-0.5 transition-colors ${stringify(demoLegs().length > 1 ? "bg-primary/10 text-primary" : "bg-surface2 text-muted-foreground")}`,
        "svelte-36n0qb"
      )}>${escape_html(demoLegs().length > 1 ? demoMode() === "hybrid" ? "fused: RRF / slider" : "fused: equal RRF" : "no fusion")}</span> <span${attr_class(
        `rounded px-2 py-0.5 transition-all ${stringify(demoSlider() ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-300" : "bg-surface2 text-muted-foreground line-through opacity-60")}`,
        "svelte-36n0qb"
      )}>Balance slider</span> <span${attr_class(
        `rounded px-2 py-0.5 transition-all ${stringify(demoRerank() ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-300" : "bg-surface2 text-muted-foreground line-through opacity-60")}`,
        "svelte-36n0qb"
      )}>Rerank</span></div> <p class="text-muted-foreground mt-3 text-xs leading-relaxed svelte-36n0qb">`);
      if (demoMode()) {
        $$renderer2.push("<!--[2-->");
        $$renderer2.push(`No image → your dial picks the mode. Toggle <strong class="text-foreground svelte-36n0qb">🖼️ Image</strong> and watch it collapse to the 4-way <code class="font-mono svelte-36n0qb">all</code>.`);
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`Add some text or an image to begin.`);
      }
      $$renderer2.push(`<!--]--></p></div> <div class="text-muted-foreground mb-1.5 text-[11px] font-medium tracking-wide uppercase svelte-36n0qb">Full reference</div> <div class="border-border overflow-x-auto rounded-lg border svelte-36n0qb"><table class="w-full border-collapse text-xs svelte-36n0qb"><thead class="svelte-36n0qb"><tr class="bg-surface2 text-foreground text-left svelte-36n0qb"><th class="border-border border-b p-2 font-medium whitespace-nowrap svelte-36n0qb">You give</th><th class="border-border border-b border-l p-2 font-medium svelte-36n0qb">Mode</th><th class="border-border border-b border-l p-2 font-medium svelte-36n0qb">Passes that run</th><th class="border-border border-b border-l p-2 font-medium svelte-36n0qb">Combined by</th><th class="border-border border-b border-l p-2 font-medium svelte-36n0qb">Slider</th><th class="border-border border-b border-l p-2 font-medium svelte-36n0qb">Rerank</th></tr></thead><tbody class="svelte-36n0qb"><!--[-->`);
      const each_array_8 = ensure_array_like(SCENARIOS);
      for (let $$index_8 = 0, $$length = each_array_8.length; $$index_8 < $$length; $$index_8++) {
        let s = each_array_8[$$index_8];
        $$renderer2.push(`<tr${attr_class(`text-muted-foreground ${stringify(s.highlight ? "bg-highlight/10" : "")}`, "svelte-36n0qb")}><td class="border-border text-foreground border-b p-2 font-medium whitespace-nowrap svelte-36n0qb">${escape_html(s.give)}</td><td class="border-border border-b border-l p-2 svelte-36n0qb"><code class="text-primary font-mono svelte-36n0qb">${escape_html(s.mode)}</code></td><td class="border-border border-b border-l p-2 svelte-36n0qb">${escape_html(s.passes)}</td><td class="border-border border-b border-l p-2 svelte-36n0qb">${escape_html(s.combined)}</td><td class="border-border border-b border-l p-2 svelte-36n0qb">${escape_html(s.slider)}</td><td class="border-border border-b border-l p-2 whitespace-nowrap svelte-36n0qb">${escape_html(s.rerank)}</td></tr>`);
      }
      $$renderer2.push(`<!--]--></tbody></table></div> <div class="reveal border-amber-500/40 bg-amber-500/5 text-muted-foreground mt-3 rounded-lg border p-3 text-sm leading-relaxed svelte-36n0qb"><strong class="text-foreground svelte-36n0qb">What adding an image does:</strong> Keyword, Vector,
          Scene, <em class="svelte-36n0qb">or</em> Hybrid all collapse to the same 4-way <code class="font-mono svelte-36n0qb">all</code> the moment you
          attach one. So <strong class="text-foreground svelte-36n0qb">Vector + image is not "just the meaning &amp; image legs"</strong> — <code class="font-mono svelte-36n0qb">all</code> runs <em class="svelte-36n0qb">four equal legs</em>, so your text <em class="svelte-36n0qb">also</em> drives a literal Keyword/BM25 leg and a Scene/caption leg. The dial you picked and the Balance
          slider stop mattering; the image is simply added as a fourth equal voice (never a gate). Image
          with no text stays a single <code class="font-mono svelte-36n0qb">visual</code> pass.</div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Worked examples: watch a full search run</h2> <p class="text-muted-foreground mb-3 text-sm leading-relaxed svelte-36n0qb">Pick a real problem and step through exactly what each stage yields.</p> <div class="reveal mb-3 rounded-lg border border-emerald-500/40 bg-emerald-500/5 p-3 text-xs leading-relaxed text-muted-foreground svelte-36n0qb"><span class="font-medium text-emerald-600 dark:text-emerald-300 svelte-36n0qb">✓ Verified live</span> against the running backend: Keyword, Vector, Hybrid, rerank, the Balance slider, Image (a posted
          frame self-matched at #1) and <code class="font-mono svelte-36n0qb">all</code> all return real
          results; a name filter returned <strong class="text-foreground svelte-36n0qb">15</strong> hits with prefilter vs <strong class="text-foreground svelte-36n0qb">0</strong> with postfilter; fuzziness 2 recovered a typo that fuzziness 0 missed. The chunk IDs
          and cosines in the examples below are illustrative — the <em class="svelte-36n0qb">mechanics</em> they show are what's
          verified.</div> <div class="mb-3 flex flex-wrap gap-1.5 svelte-36n0qb"><!--[-->`);
      const each_array_9 = ensure_array_like(WORKED);
      for (let i = 0, $$length = each_array_9.length; i < $$length; i++) {
        let w = each_array_9[i];
        $$renderer2.push(`<button type="button"${attr_class(
          `rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors ${stringify(exIdx === i ? "border-primary bg-primary/10 text-primary" : "border-border text-muted-foreground hover:text-foreground")}`,
          "svelte-36n0qb"
        )}>${escape_html(w.tab)}</button>`);
      }
      $$renderer2.push(`<!--]--></div> <!---->`);
      {
        const w = WORKED[exIdx];
        $$renderer2.push(`<div class="bg-card border-border rounded-xl border p-4 svelte-36n0qb"><div class="flex flex-wrap items-center gap-2 svelte-36n0qb"><span class="text-foreground text-sm font-semibold svelte-36n0qb">${escape_html(w.title)}</span> <code class="bg-primary/15 text-primary rounded px-2 py-0.5 font-mono text-[11px] font-semibold svelte-36n0qb">mode: ${escape_html(w.mode)}</code></div> <p class="text-muted-foreground mt-1 text-xs leading-relaxed svelte-36n0qb">${escape_html(w.why)}</p> <ol class="mt-4 space-y-3 svelte-36n0qb"><!--[-->`);
        const each_array_10 = ensure_array_like(w.steps);
        for (let i = 0, $$length = each_array_10.length; i < $$length; i++) {
          let s = each_array_10[i];
          $$renderer2.push(`<li class="reveal flex gap-3 svelte-36n0qb"><span class="bg-primary/15 text-primary grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold svelte-36n0qb">${escape_html(i + 1)}</span> <div class="min-w-0 svelte-36n0qb"><div class="text-foreground text-xs font-semibold svelte-36n0qb">${escape_html(s.stage)}</div> <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed [overflow-wrap:anywhere] svelte-36n0qb">${escape_html(s.yields)}</p></div></li>`);
        }
        $$renderer2.push(`<!--]--></ol> <div class="reveal border-primary/40 bg-primary/5 mt-4 rounded-lg border p-3 text-xs leading-relaxed svelte-36n0qb"><span class="text-foreground font-semibold svelte-36n0qb">Result:</span> <span class="text-muted-foreground svelte-36n0qb">${escape_html(w.result)}</span></div> <div class="reveal bg-surface2 border-border mt-2 rounded-lg border p-3 text-xs leading-relaxed svelte-36n0qb"><span class="text-foreground font-semibold svelte-36n0qb">Takeaway:</span> <span class="text-muted-foreground svelte-36n0qb">${escape_html(w.takeaway)}</span></div></div>`);
      }
      $$renderer2.push(`<!----></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-4 text-lg font-semibold svelte-36n0qb">The whole pipeline</h2> <div class="flex flex-col items-center svelte-36n0qb"><div class="grid w-full grid-cols-3 gap-2 svelte-36n0qb"><div class="bg-surface2 border-border border-l-[3px] border-l-sky-400 rounded-md border px-2 py-2 text-center text-xs svelte-36n0qb">⌨️ Keyword text</div> <div class="bg-surface2 border-border border-l-[3px] border-l-primary rounded-md border px-2 py-2 text-center text-xs svelte-36n0qb">💬 Meaning text</div> <div class="bg-surface2 border-border border-l-[3px] border-l-amber-400 rounded-md border px-2 py-2 text-center text-xs svelte-36n0qb">🖼️ Image</div></div> <div class="text-muted-foreground grid w-full grid-cols-3 text-center text-[10px] svelte-36n0qb"><span class="svelte-36n0qb">▼</span><span class="svelte-36n0qb">▼ embed</span><span class="svelte-36n0qb">▼ embed</span></div> <div class="grid w-full grid-cols-3 gap-2 svelte-36n0qb"><div class="bg-background border-border rounded-md border px-2 py-1.5 text-center text-[11px] svelte-36n0qb">FTS leg <div class="text-muted-foreground svelte-36n0qb">BM25 on text</div></div> <div class="bg-background border-border rounded-md border px-2 py-1.5 text-center text-[11px] svelte-36n0qb">Text-vector leg <div class="text-muted-foreground svelte-36n0qb">text_embedding</div></div> <div class="bg-background border-border rounded-md border px-2 py-1.5 text-center text-[11px] svelte-36n0qb">Frame-vector leg <div class="text-muted-foreground svelte-36n0qb">frame_embedding</div></div></div> <p class="text-muted-foreground mt-2 text-center text-[11px] svelte-36n0qb">+ the text query also drives a <strong class="text-foreground svelte-36n0qb">Scene</strong> leg (<code class="font-mono svelte-36n0qb">caption_embedding</code>) in <code class="font-mono svelte-36n0qb">all</code>.</p> <div class="flow-line svelte-36n0qb"><span class="flow-dot svelte-36n0qb"></span></div> <div class="border-primary/50 bg-primary/10 text-foreground w-full rounded-lg border px-4 py-2.5 text-center text-sm font-medium svelte-36n0qb">Fuse into one ranking — <span class="font-mono svelte-36n0qb">RRF</span> (default) or Balance slider (hybrid
            only)</div> <div class="flow-line svelte-36n0qb"><span class="flow-dot svelte-36n0qb" style="animation-delay:.3s"></span></div> <div class="bg-surface2 border-border w-full rounded-md border px-4 py-2 text-center text-sm svelte-36n0qb">Rerank top <em class="svelte-36n0qb">K</em> <span class="text-muted-foreground svelte-36n0qb">(optional)</span> — re-read each
            transcript vs your text</div> <div class="flow-line svelte-36n0qb"><span class="flow-dot svelte-36n0qb" style="animation-delay:.6s"></span></div> <div class="bg-card border-border text-foreground w-full rounded-lg border px-4 py-2.5 text-center font-semibold svelte-36n0qb">Top N ranked chunk results</div></div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">How the lists merge: RRF (worked example)</h2> <p class="text-muted-foreground mb-3 text-sm leading-relaxed svelte-36n0qb">Think of every active signal as a judge handing in a ranked shortlist. The clip that lands
          high on the <strong class="text-foreground svelte-36n0qb">most</strong> shortlists wins — even if it's nobody's outright
          #1. Here it is with two judges on 5 clips (A–E), each returning a ranked list:</p> <div class="grid gap-3 sm:grid-cols-2 svelte-36n0qb"><div class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs svelte-36n0qb"><div class="text-foreground mb-1 font-medium svelte-36n0qb">⌨️ Keyword</div> <div class="text-muted-foreground svelte-36n0qb"><!--[-->`);
      const each_array_11 = ensure_array_like(keywordRanked);
      for (let i = 0, $$length = each_array_11.length; i < $$length; i++) {
        let c = each_array_11[i];
        $$renderer2.push(`<span class="text-foreground font-mono svelte-36n0qb">${escape_html(i + 1)} = ${escape_html(c)}</span>`);
        if (i < keywordRanked.length - 1) {
          $$renderer2.push("<!--[0-->");
          $$renderer2.push(` · `);
        } else {
          $$renderer2.push("<!--[-1-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--></div></div> <div class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs svelte-36n0qb"><div class="text-foreground mb-1 font-medium svelte-36n0qb">💬 Meaning</div> <div class="text-muted-foreground svelte-36n0qb"><!--[-->`);
      const each_array_12 = ensure_array_like(meaningRanked);
      for (let i = 0, $$length = each_array_12.length; i < $$length; i++) {
        let c = each_array_12[i];
        $$renderer2.push(`<span class="text-foreground font-mono svelte-36n0qb">${escape_html(i + 1)} = ${escape_html(c)}</span>`);
        if (i < meaningRanked.length - 1) {
          $$renderer2.push("<!--[0-->");
          $$renderer2.push(` · `);
        } else {
          $$renderer2.push("<!--[-1-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--></div></div></div> <p class="text-muted-foreground mt-3 mb-2 text-sm svelte-36n0qb">Each clip scores <code class="text-primary bg-surface2 rounded px-1 font-mono text-[11px] svelte-36n0qb">Σ 1/(60 + rank)</code> over the lists it appears in (rank counts from 0, so the #1 clip uses 1/60):</p> <div class="bg-card border-border overflow-hidden rounded-lg border svelte-36n0qb"><!--[-->`);
      const each_array_13 = ensure_array_like(rrfRows);
      for (let i = 0, $$length = each_array_13.length; i < $$length; i++) {
        let row = each_array_13[i];
        $$renderer2.push(`<div${attr_class(`reveal border-border flex items-center gap-3 border-b px-3 py-2 text-xs last:border-b-0 ${stringify(row.inBoth ? "bg-highlight/10" : "")}`, "svelte-36n0qb")}><span class="text-foreground w-6 font-mono font-medium svelte-36n0qb">${escape_html(row.clip)}</span> <span class="text-muted-foreground flex-1 font-mono svelte-36n0qb">${escape_html(row.parts.map((p) => `1/${K + p.rank}`).join(" + "))}</span> <span class="text-foreground font-mono svelte-36n0qb">≈ ${escape_html(row.score.toFixed(4))}</span> `);
        if (row.inBoth) {
          $$renderer2.push("<!--[0-->");
          $$renderer2.push(`<span class="bg-highlight/30 text-foreground rounded px-1.5 py-0.5 text-[10px] font-medium svelte-36n0qb">in both → wins</span>`);
        } else {
          $$renderer2.push("<!--[-1-->");
          $$renderer2.push(`<span class="text-muted-foreground text-[10px] svelte-36n0qb">one list</span>`);
        }
        $$renderer2.push(`<!--]--></div>`);
      }
      $$renderer2.push(`<!--]--></div> <p class="text-muted-foreground mt-3 text-sm leading-relaxed svelte-36n0qb">Clips that appear in <strong class="text-foreground svelte-36n0qb">more lists</strong> rise to the top.
          RRF needs no tuning and works the same for <strong class="text-foreground svelte-36n0qb">2, 3, or 4 lists</strong> — you just add another <code class="font-mono svelte-36n0qb">1/(60 + rank)</code> term. The multi-judge <code class="font-mono svelte-36n0qb">all</code> mode <em class="svelte-36n0qb">always</em> uses equal-weight RRF.</p> <div class="reveal border-border bg-card mt-3 rounded-lg border p-3 text-xs leading-relaxed text-muted-foreground svelte-36n0qb"><div class="text-foreground mb-1 font-medium svelte-36n0qb">How the merge actually works (and why any number of legs is fine)</div> The legs run<strong class="text-foreground svelte-36n0qb">independently</strong> — none feeds its hits
          to another (the image's results are <em class="svelte-36n0qb">not</em> handed to the Vector leg). Each returns
          its <em class="svelte-36n0qb">own</em> ranked list over the whole library. RRF then pours all the lists into <strong class="text-foreground svelte-36n0qb">one pot keyed by chunk</strong>: a chunk's score is the <strong class="text-foreground svelte-36n0qb">sum</strong> of <code class="font-mono svelte-36n0qb">1/(60 + rank)</code> from <em class="svelte-36n0qb">every</em> list it appears in
          (absent from a list → adds 0). So it's a <strong class="text-foreground svelte-36n0qb">union with added scores</strong> — not a chain, and not a plain concatenation. Because it's just a sum, it extends to <strong class="text-foreground svelte-36n0qb">any number of lists</strong> (2 for Hybrid, up to 4 for <code class="font-mono svelte-36n0qb">all</code>) — you add one more term per
          leg. A chunk only one leg found still shows up (it just scores low); a chunk several legs
          rank highly wins.</div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Balance slider vs RRF — the key difference</h2> <p class="text-muted-foreground mb-3 text-sm leading-relaxed svelte-36n0qb">The Balance slider is a <strong class="text-foreground svelte-36n0qb">2-way blend</strong> of the actual
          scores; it only exists for Hybrid (keyword ↔ meaning). It cannot describe 3 legs, so the
          moment you add an image, fusion falls back to equal-weight RRF and the slider is <strong class="text-foreground svelte-36n0qb">ignored</strong>.</p> <div class="reveal bg-surface2 border-border mb-3 rounded-lg border p-3 text-xs leading-relaxed svelte-36n0qb"><span class="text-foreground font-medium svelte-36n0qb">For instance:</span> searching <code class="bg-card text-primary rounded px-1 font-mono svelte-36n0qb">elcertifikat</code> → slide
          toward <strong class="text-foreground svelte-36n0qb">Keyword</strong> for that exact word; searching <code class="bg-card text-primary rounded px-1 font-mono svelte-36n0qb">vad regeringen gör åt höga elpriser</code> → slide toward <strong class="text-foreground svelte-36n0qb">Vector</strong> to catch a clip that says
          "stötta hushållen med elkostnaderna". The slider only re-weights the two <em class="svelte-36n0qb">text</em> passes
          — a weighted blend of their scores, not rank-voting — and does nothing for Scene or Image.</div> <div class="grid gap-3 sm:grid-cols-2 svelte-36n0qb"><div class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs svelte-36n0qb"><div class="text-foreground font-medium svelte-36n0qb">Slider (2 judges only)</div> <div class="text-muted-foreground mt-0.5 svelte-36n0qb">Hybrid keyword ↔ meaning.</div> <div class="text-foreground mt-2 font-mono svelte-36n0qb">w·vectorScore + (1−w)·ftsScore</div> <div class="text-muted-foreground mt-0.5 svelte-36n0qb">Uses real scores. You pick the weight.</div></div> <div class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs svelte-36n0qb"><div class="text-foreground font-medium svelte-36n0qb">RRF (2, 3 or 4 judges)</div> <div class="text-muted-foreground mt-0.5 svelte-36n0qb">Default everywhere; always for multi-leg "all".</div> <div class="text-foreground mt-2 font-mono svelte-36n0qb">Σ 1/(60 + rank)</div> <div class="text-muted-foreground mt-0.5 svelte-36n0qb">Uses ranks only. Equal weight, no tuning.</div></div></div></section> <section class="reveal bg-card border-border mb-10 rounded-lg border p-4 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Rerank — what it does, and which column it reads</h2> <p class="text-muted-foreground mb-3 text-sm leading-relaxed svelte-36n0qb">A second, slower-but-sharper precision pass. A cross-encoder reads your query text and one
          chunk's transcript <em class="svelte-36n0qb">together</em> and scores how well they actually answer each other — better
          than the first-stage ranking, so it only runs on the head.</p> <div class="border-border overflow-hidden rounded-lg border text-xs svelte-36n0qb"><div class="text-muted-foreground grid grid-cols-[7.5rem_1fr] svelte-36n0qb"><div class="border-border text-foreground border-b p-2 font-medium svelte-36n0qb">Reads</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">one column — each candidate's transcript, the <code class="bg-surface2 text-primary rounded px-1 font-mono svelte-36n0qb">text</code> field on the chunk. Nothing else.</div></div> <div class="text-muted-foreground grid grid-cols-[7.5rem_1fr] svelte-36n0qb"><div class="border-border text-foreground border-b p-2 font-medium svelte-36n0qb">Scores against</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">your <code class="font-mono svelte-36n0qb">Keyword + Meaning</code> text joined (<code class="font-mono svelte-36n0qb">q + q_vec</code>).</div></div> <div class="text-muted-foreground grid grid-cols-[7.5rem_1fr] svelte-36n0qb"><div class="border-border text-foreground border-b p-2 font-medium svelte-36n0qb">Scope</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">the top <strong class="text-foreground svelte-36n0qb">K</strong> (default 20) of the fused list; it reorders
              just that head. Everything below K keeps its first-stage order — no new search runs.</div></div> <div class="text-muted-foreground grid grid-cols-[7.5rem_1fr] svelte-36n0qb"><div class="text-foreground p-2 font-medium svelte-36n0qb">Ignores</div> <div class="border-border border-l p-2 svelte-36n0qb">the image, the frame <code class="font-mono svelte-36n0qb">caption</code>, and <strong class="text-foreground svelte-36n0qb">all</strong> vectors. So an image-only search (no query text) → rerank is a no-op.</div></div></div> <div class="reveal bg-surface2 border-border mt-3 rounded-lg border p-3 text-xs leading-relaxed svelte-36n0qb"><span class="text-foreground font-medium svelte-36n0qb">Before → after</span>, query <code class="bg-card text-primary rounded px-1 font-mono svelte-36n0qb">höjda räntor</code>: the fused
          top-3 might be ① a "minister vid podium" clip that drifted in on Vector, ② a stakeout that
          says "räntan" once, ③ the clip that actually says "vi måste hantera de höjda räntorna".
          The cross-encoder reads only each transcript and reorders to <strong class="text-foreground svelte-36n0qb">③ ① ②</strong> — promoting the clip whose spoken words really
          match.</div></section> <section class="reveal border-border bg-surface2 mb-10 rounded-lg border p-4 svelte-36n0qb"><h2 class="text-foreground mb-2 text-lg font-semibold svelte-36n0qb">The one real "narrow-first": Filters</h2> <p class="text-muted-foreground text-sm leading-relaxed svelte-36n0qb">The <strong class="text-foreground svelte-36n0qb">Filters</strong> panel (language, name, referenskod,
          extraid, raw SQL) is the only thing that narrows the library — and it has nothing to do
          with the image. A filter becomes a <code class="font-mono svelte-36n0qb">WHERE</code> clause. For the <strong class="text-foreground svelte-36n0qb">transcript legs</strong> (Keyword, Vector, Hybrid), <em class="svelte-36n0qb">prefilter</em> narrows the corpus before they search (off → after); the <strong class="text-foreground svelte-36n0qb">frame legs</strong> (Image, Scene) always filter after ranking. So <strong class="text-foreground svelte-36n0qb">Filter = a fence around the whole library</strong>; an
          attached image = just one more voice in the merge.</p> <div class="reveal border-border mt-3 overflow-hidden rounded-lg border text-xs svelte-36n0qb"><div class="bg-surface2 text-foreground grid grid-cols-[1fr_1.5fr] font-medium svelte-36n0qb"><div class="border-border border-b p-2 svelte-36n0qb">Filter</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">What it targets (all on the chunk's metadata)</div></div> <div class="text-muted-foreground grid grid-cols-[1fr_1.5fr] svelte-36n0qb"><div class="border-border text-foreground border-b p-2 svelte-36n0qb">Language</div> <div class="border-border border-b border-l p-2 svelte-36n0qb"><code class="font-mono svelte-36n0qb">language = …</code> — exact match</div></div> <div class="text-muted-foreground grid grid-cols-[1fr_1.5fr] svelte-36n0qb"><div class="border-border text-foreground border-b p-2 svelte-36n0qb">Name (namn)</div> <div class="border-border border-b border-l p-2 svelte-36n0qb"><code class="font-mono svelte-36n0qb">namn LIKE '%…%'</code> — substring match</div></div> <div class="text-muted-foreground grid grid-cols-[1fr_1.5fr] svelte-36n0qb"><div class="border-border text-foreground border-b p-2 svelte-36n0qb">Ref (referenskod)</div> <div class="border-border border-b border-l p-2 svelte-36n0qb"><code class="font-mono svelte-36n0qb">referenskod LIKE '%…%'</code> — substring match</div></div> <div class="text-muted-foreground grid grid-cols-[1fr_1.5fr] svelte-36n0qb"><div class="border-border text-foreground border-b p-2 svelte-36n0qb">Internal ID (extraid)</div> <div class="border-border border-b border-l p-2 svelte-36n0qb"><code class="font-mono svelte-36n0qb">extraid = …</code> — exact match</div></div> <div class="text-muted-foreground grid grid-cols-[1fr_1.5fr] svelte-36n0qb"><div class="border-border text-foreground p-2 svelte-36n0qb">Raw SQL</div> <div class="border-border border-l p-2 svelte-36n0qb">any expression over chunk columns, ANDed in verbatim</div></div></div> <p class="reveal text-muted-foreground mt-2 text-xs leading-relaxed svelte-36n0qb">All conditions are <strong class="text-foreground svelte-36n0qb">ANDed</strong> together and target the <code class="font-mono svelte-36n0qb">chunks</code> table's metadata — <em class="svelte-36n0qb">not</em> frame attributes —
          so even Image and Scene results are filtered by <em class="svelte-36n0qb">their chunk's</em> language / name / ref
          / id.</p> <div class="reveal bg-card border-border mt-3 rounded-lg border p-3 text-xs leading-relaxed svelte-36n0qb"><span class="text-foreground font-medium svelte-36n0qb">For instance:</span> search <code class="bg-surface2 text-primary rounded px-1 font-mono svelte-36n0qb">nya regler</code> with a
          filter <code class="bg-surface2 rounded px-1 font-mono svelte-36n0qb">namn = "Andersson"</code> and N = 10. <strong class="text-foreground svelte-36n0qb">Prefilter (on):</strong> every leg that runs (here Keyword
          + Vector) searches <em class="svelte-36n0qb">only</em> Andersson's chunks → a full 10 ranked Andersson hits. <strong class="text-foreground svelte-36n0qb">Postfilter (off):</strong> each leg first finds its top 10
          across <em class="svelte-36n0qb">all</em> ministers, then drops the non-Andersson ones — if only 3 were Andersson, you see
          3. A tight postfilter can come up short.</div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Reading &amp; fixing your results</h2> <p class="text-muted-foreground mb-4 text-sm leading-relaxed svelte-36n0qb">Every hit won a popularity contest among the signals you turned on. Read a surprising
          result by asking "which side did it win on?", and fix a bad list by changing <em class="svelte-36n0qb">who votes</em>, not by staring harder. Mental dial: <strong class="text-foreground svelte-36n0qb">recall vs precision</strong> — widen when you're missing things,
          tighten when the top is junk.</p> <div class="border-border overflow-hidden rounded-lg border text-xs svelte-36n0qb"><!--[-->`);
      const each_array_14 = ensure_array_like(TROUBLESHOOT);
      for (let i = 0, $$length = each_array_14.length; i < $$length; i++) {
        let r = each_array_14[i];
        $$renderer2.push(`<div class="reveal grid grid-cols-[1fr_1.5fr] last:[&amp;>div]:border-b-0 svelte-36n0qb"><div class="border-border text-foreground border-b p-2 font-medium svelte-36n0qb">${escape_html(r.symptom)}</div> <div class="border-border text-muted-foreground border-b border-l p-2 svelte-36n0qb">${escape_html(r.fix)}</div></div>`);
      }
      $$renderer2.push(`<!--]--></div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Under the hood: how a query travels</h2> <p class="text-muted-foreground mb-4 text-sm leading-relaxed svelte-36n0qb">What actually happens between hitting Enter and seeing results.</p> <ol class="space-y-3 svelte-36n0qb"><!--[-->`);
      const each_array_15 = ensure_array_like(JOURNEY);
      for (let i = 0, $$length = each_array_15.length; i < $$length; i++) {
        let s = each_array_15[i];
        $$renderer2.push(`<li class="reveal flex gap-3 svelte-36n0qb"><span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold svelte-36n0qb">${escape_html(i + 1)}</span> <div class="svelte-36n0qb"><div class="text-foreground text-sm font-medium svelte-36n0qb">${escape_html(s.label)}</div> <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed svelte-36n0qb">${escape_html(s.detail)}</p></div></li>`);
      }
      $$renderer2.push(`<!--]--></ol> <div class="mt-4 grid gap-3 sm:grid-cols-2 svelte-36n0qb"><div class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs leading-relaxed svelte-36n0qb"><div class="text-foreground font-medium svelte-36n0qb">Two tables, one key</div> <p class="text-muted-foreground mt-1 svelte-36n0qb"><code class="font-mono svelte-36n0qb">chunks</code> holds the transcript side; <code class="font-mono svelte-36n0qb">chunk_frames</code> holds one row per video frame (its caption + image vector). They're linked by <code class="font-mono svelte-36n0qb">(doc_id, speech_id, chunk_id)</code>. Image / Scene search the
              frames table, then join back — which is why every result is a <em class="svelte-36n0qb">chunk</em>, even
              when the match was visual.</p></div> <div class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs leading-relaxed svelte-36n0qb"><div class="text-foreground font-medium svelte-36n0qb">One shared embedding space</div> <p class="text-muted-foreground mt-1 svelte-36n0qb">Transcript and caption vectors live in the <em class="svelte-36n0qb">same</em> space, so a single query-text vector
              can be compared to both (Vector → transcripts, Scene → captions). Your image vector lives
              with the frame vectors (Image). Same map, different pins.</p></div> <div class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs leading-relaxed sm:col-span-2 svelte-36n0qb"><div class="text-foreground font-medium svelte-36n0qb">Where the work happens</div> <p class="text-muted-foreground mt-1 svelte-36n0qb">The <strong class="text-foreground svelte-36n0qb">frontend</strong> (this UI) talks to the <strong class="text-foreground svelte-36n0qb">backend</strong>, which talks to two separate model
              servers — one for <strong class="text-foreground svelte-36n0qb">embeddings</strong>, one for <strong class="text-foreground svelte-36n0qb">reranking</strong> — and reads the LanceDB tables on
              disk. If the embedding server is down a vector search fails fast with a clear error;
              if an embedding column was never built, that leg returns nothing in a fused search and
              the others carry on (a single Vector or Hybrid search instead shows a clear "not
              built" message). The <strong class="text-foreground svelte-36n0qb">Services</strong> badge in the sidebar shows their live
              status.</p></div></div></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-3 text-lg font-semibold svelte-36n0qb">⚙️ Settings reference</h2> <div class="border-border overflow-hidden rounded-lg border text-xs svelte-36n0qb"><!--[-->`);
      const each_array_16 = ensure_array_like(SETTINGS);
      for (let i = 0, $$length = each_array_16.length; i < $$length; i++) {
        let s = each_array_16[i];
        $$renderer2.push(`<div class="reveal text-muted-foreground grid grid-cols-[1fr_1.7fr] last:[&amp;>div]:border-b-0 svelte-36n0qb"><div class="border-border text-foreground border-b p-2 svelte-36n0qb">${escape_html(s.name)}</div> <div class="border-border border-b border-l p-2 svelte-36n0qb">${escape_html(s.desc)}</div></div>`);
      }
      $$renderer2.push(`<!--]--></div></section> <section class="reveal bg-surface2 border-border mb-8 rounded-lg border p-4 svelte-36n0qb"><h2 class="text-foreground mb-2 text-lg font-semibold svelte-36n0qb">Known limitations</h2> <ul class="text-muted-foreground list-disc space-y-1 pl-5 text-sm leading-relaxed svelte-36n0qb"><!--[-->`);
      const each_array_17 = ensure_array_like(LIMITS);
      for (let $$index_17 = 0, $$length = each_array_17.length; $$index_17 < $$length; $$index_17++) {
        let l = each_array_17[$$index_17];
        $$renderer2.push(`<li class="svelte-36n0qb">${escape_html(l)}</li>`);
      }
      $$renderer2.push(`<!--]--></ul></section> <section class="reveal mb-10 svelte-36n0qb"><h2 class="text-foreground mb-1 text-lg font-semibold svelte-36n0qb">Glossary — the words you'll see</h2> <p class="text-muted-foreground mb-4 text-sm leading-relaxed svelte-36n0qb">Each term in one line. None of them are magic — each is a librarian with one specific
          talent.</p> <dl class="border-border divide-border divide-y overflow-hidden rounded-lg border text-xs svelte-36n0qb"><!--[-->`);
      const each_array_18 = ensure_array_like(GLOSSARY);
      for (let i = 0, $$length = each_array_18.length; i < $$length; i++) {
        let g = each_array_18[i];
        $$renderer2.push(`<div class="reveal grid grid-cols-[8.5rem_1fr] gap-2 p-2 svelte-36n0qb"><dt class="text-foreground font-medium svelte-36n0qb">${escape_html(g.term)}</dt> <dd class="text-muted-foreground svelte-36n0qb">${escape_html(g.def)}</dd></div>`);
      }
      $$renderer2.push(`<!--]--></dl></section> <a href="/" class="reveal border-primary/40 bg-primary/10 text-foreground hover:bg-primary/15 group flex items-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition-colors svelte-36n0qb">Try it on the Search page `);
      Arrow_right($$renderer2, {
        class: "size-4 transition-transform group-hover:translate-x-0.5"
      });
      $$renderer2.push(`<!----></a>`);
    }
    $$renderer2.push(`<!--]--></div></div>`);
  });
}
export {
  _page as default
};
