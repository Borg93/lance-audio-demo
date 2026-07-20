import "../../chunks/async.js";
import { s as sanitize_props, a as spread_props, b as slot, e as attributes, f as clsx, j as bind_props, d as derived, p as props_id, i as attr_class, m as escape_html, l as ensure_array_like, h as attr, k as stringify, n as attr_style } from "../../chunks/renderer.js";
import "../../chunks/client.js";
import { b as getVoiceStatus, c as activeView, t as thumbnailUrl, d as chunkFrameUrl, i as isVoiceHit, v as voiceBandOf, r as relevanceOf, e as search, A as ApiError, f as voiceSimilarUpload, h as voiceSimilar, j as getAtlasChunks } from "../../chunks/api.js";
import "clsx";
import { c as cn, i as isWritableSymbol, a as BoxSymbol, b as boxFrom, d as boxWith, e as boxFlatten, t as toReadonlyBox, f as isBox, g as isWritableBox, A as ARROW_UP, h as ARROW_RIGHT, j as ARROW_LEFT, k as ARROW_DOWN, E as END, l as isHTMLElement, H as HOME, C as Context, m as attachRef, n as boolToEmptyStrOrUndef, o as boolToStr, w as watch, S as SPACE, p as getAriaChecked, q as createBitsAttrs, r as createId, s as noop, u as mergeProps, v as isElementOrSVGElement, D as DOMContext, x as ENTER, y as getDataChecked, z as boolToTrueOrUndef, P as Portal, B as Button, F as hitKey, G as fmtTime, I as makeHighlighter, J as queryTerms } from "../../chunks/scroll-lock.js";
import { H as Hidden_input, i as isValidIndex, S as Select_1 } from "../../chunks/select.js";
import { P as Popover, a as Popover_trigger, b as Popover_content, A as Audio_lines } from "../../chunks/popover.js";
import { I as Icon } from "../../chunks/Icon.js";
import { X, P as Plus, L as Loader_circle, a as Play, R as Resizable_split, b as Player_pane } from "../../chunks/resizable-split.js";
import { A as Arrow_right } from "../../chunks/arrow-right.js";
import { r as run } from "../../chunks/render-context.js";
import { S as Search } from "../../chunks/search.js";
import "@sveltejs/kit/internal";
import "../../chunks/exports.js";
import "../../chunks/utils.js";
import "@sveltejs/kit/internal/server";
import "../../chunks/root.js";
import "../../chunks/state.svelte.js";
import { S as SvelteSet } from "../../chunks/index-server.js";
import * as v from "valibot";
import { M as Map$1 } from "../../chunks/map.js";
import { C as Chevron_right } from "../../chunks/sr-only-styles.js";
function html(value) {
  var html2 = String(value ?? "");
  var open = "<!---->";
  return open + html2 + "<!---->";
}
function Chevron_left($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [["path", { "d": "m15 18-6-6 6-6" }]];
  Icon($$renderer, spread_props([
    { name: "chevron-left" },
    $$sanitized_props,
    {
      /**
       * @component @name ChevronLeft
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJtMTUgMTgtNi02IDYtNiIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/chevron-left
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
function Circle_help($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["circle", { "cx": "12", "cy": "12", "r": "10" }],
    ["path", { "d": "M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" }],
    ["path", { "d": "M12 17h.01" }]
  ];
  Icon($$renderer, spread_props([
    { name: "circle-help" },
    $$sanitized_props,
    {
      /**
       * @component @name CircleHelp
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgLz4KICA8cGF0aCBkPSJNOS4wOSA5YTMgMyAwIDAgMSA1LjgzIDFjMCAyLTMgMy0zIDMiIC8+CiAgPHBhdGggZD0iTTEyIDE3aC4wMSIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/circle-help
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
function Filter($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "polygon",
      { "points": "22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "filter" },
    $$sanitized_props,
    {
      /**
       * @component @name Filter
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cG9seWdvbiBwb2ludHM9IjIyIDMgMiAzIDEwIDEyLjQ2IDEwIDE5IDE0IDIxIDE0IDEyLjQ2IDIyIDMiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/filter
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
function Image_plus($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M16 5h6" }],
    ["path", { "d": "M19 2v6" }],
    [
      "path",
      {
        "d": "M21 11.5V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7.5"
      }
    ],
    ["path", { "d": "m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" }],
    ["circle", { "cx": "9", "cy": "9", "r": "2" }]
  ];
  Icon($$renderer, spread_props([
    { name: "image-plus" },
    $$sanitized_props,
    {
      /**
       * @component @name ImagePlus
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTYgNWg2IiAvPgogIDxwYXRoIGQ9Ik0xOSAydjYiIC8+CiAgPHBhdGggZD0iTTIxIDExLjVWMTlhMiAyIDAgMCAxLTIgMkg1YTIgMiAwIDAgMS0yLTJWNWEyIDIgMCAwIDEgMi0yaDcuNSIgLz4KICA8cGF0aCBkPSJtMjEgMTUtMy4wODYtMy4wODZhMiAyIDAgMCAwLTIuODI4IDBMNiAyMSIgLz4KICA8Y2lyY2xlIGN4PSI5IiBjeT0iOSIgcj0iMiIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/image-plus
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
function Image($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "rect",
      {
        "width": "18",
        "height": "18",
        "x": "3",
        "y": "3",
        "rx": "2",
        "ry": "2"
      }
    ],
    ["circle", { "cx": "9", "cy": "9", "r": "2" }],
    ["path", { "d": "m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" }]
  ];
  Icon($$renderer, spread_props([
    { name: "image" },
    $$sanitized_props,
    {
      /**
       * @component @name Image
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cmVjdCB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHg9IjMiIHk9IjMiIHJ4PSIyIiByeT0iMiIgLz4KICA8Y2lyY2xlIGN4PSI5IiBjeT0iOSIgcj0iMiIgLz4KICA8cGF0aCBkPSJtMjEgMTUtMy4wODYtMy4wODZhMiAyIDAgMCAwLTIuODI4IDBMNiAyMSIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/image
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
function Layout_grid($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "rect",
      { "width": "7", "height": "7", "x": "3", "y": "3", "rx": "1" }
    ],
    [
      "rect",
      { "width": "7", "height": "7", "x": "14", "y": "3", "rx": "1" }
    ],
    [
      "rect",
      { "width": "7", "height": "7", "x": "14", "y": "14", "rx": "1" }
    ],
    [
      "rect",
      { "width": "7", "height": "7", "x": "3", "y": "14", "rx": "1" }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "layout-grid" },
    $$sanitized_props,
    {
      /**
       * @component @name LayoutGrid
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cmVjdCB3aWR0aD0iNyIgaGVpZ2h0PSI3IiB4PSIzIiB5PSIzIiByeD0iMSIgLz4KICA8cmVjdCB3aWR0aD0iNyIgaGVpZ2h0PSI3IiB4PSIxNCIgeT0iMyIgcng9IjEiIC8+CiAgPHJlY3Qgd2lkdGg9IjciIGhlaWdodD0iNyIgeD0iMTQiIHk9IjE0IiByeD0iMSIgLz4KICA8cmVjdCB3aWR0aD0iNyIgaGVpZ2h0PSI3IiB4PSIzIiB5PSIxNCIgcng9IjEiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/layout-grid
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
function List($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M3 12h.01" }],
    ["path", { "d": "M3 18h.01" }],
    ["path", { "d": "M3 6h.01" }],
    ["path", { "d": "M8 12h13" }],
    ["path", { "d": "M8 18h13" }],
    ["path", { "d": "M8 6h13" }]
  ];
  Icon($$renderer, spread_props([
    { name: "list" },
    $$sanitized_props,
    {
      /**
       * @component @name List
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMyAxMmguMDEiIC8+CiAgPHBhdGggZD0iTTMgMThoLjAxIiAvPgogIDxwYXRoIGQ9Ik0zIDZoLjAxIiAvPgogIDxwYXRoIGQ9Ik04IDEyaDEzIiAvPgogIDxwYXRoIGQ9Ik04IDE4aDEzIiAvPgogIDxwYXRoIGQ9Ik04IDZoMTMiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/list
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
function Minus($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [["path", { "d": "M5 12h14" }]];
  Icon($$renderer, spread_props([
    { name: "minus" },
    $$sanitized_props,
    {
      /**
       * @component @name Minus
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNNSAxMmgxNCIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/minus
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
function Paperclip($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M13.234 20.252 21 12.3" }],
    [
      "path",
      {
        "d": "m16 6-8.414 8.586a2 2 0 0 0 0 2.828 2 2 0 0 0 2.828 0l8.414-8.586a4 4 0 0 0 0-5.656 4 4 0 0 0-5.656 0l-8.415 8.585a6 6 0 1 0 8.486 8.486"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "paperclip" },
    $$sanitized_props,
    {
      /**
       * @component @name Paperclip
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTMuMjM0IDIwLjI1MiAyMSAxMi4zIiAvPgogIDxwYXRoIGQ9Im0xNiA2LTguNDE0IDguNTg2YTIgMiAwIDAgMCAwIDIuODI4IDIgMiAwIDAgMCAyLjgyOCAwbDguNDE0LTguNTg2YTQgNCAwIDAgMCAwLTUuNjU2IDQgNCAwIDAgMC01LjY1NiAwbC04LjQxNSA4LjU4NWE2IDYgMCAxIDAgOC40ODYgOC40ODYiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/paperclip
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
function Pause($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "rect",
      { "x": "14", "y": "4", "width": "4", "height": "16", "rx": "1" }
    ],
    [
      "rect",
      { "x": "6", "y": "4", "width": "4", "height": "16", "rx": "1" }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "pause" },
    $$sanitized_props,
    {
      /**
       * @component @name Pause
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cmVjdCB4PSIxNCIgeT0iNCIgd2lkdGg9IjQiIGhlaWdodD0iMTYiIHJ4PSIxIiAvPgogIDxyZWN0IHg9IjYiIHk9IjQiIHdpZHRoPSI0IiBoZWlnaHQ9IjE2IiByeD0iMSIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/pause
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
function Search_x($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "m13.5 8.5-5 5" }],
    ["path", { "d": "m8.5 8.5 5 5" }],
    ["circle", { "cx": "11", "cy": "11", "r": "8" }],
    ["path", { "d": "m21 21-4.3-4.3" }]
  ];
  Icon($$renderer, spread_props([
    { name: "search-x" },
    $$sanitized_props,
    {
      /**
       * @component @name SearchX
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJtMTMuNSA4LjUtNSA1IiAvPgogIDxwYXRoIGQ9Im04LjUgOC41IDUgNSIgLz4KICA8Y2lyY2xlIGN4PSIxMSIgY3k9IjExIiByPSI4IiAvPgogIDxwYXRoIGQ9Im0yMSAyMS00LjMtNC4zIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/search-x
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
function Settings_2($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M20 7h-9" }],
    ["path", { "d": "M14 17H5" }],
    ["circle", { "cx": "17", "cy": "17", "r": "3" }],
    ["circle", { "cx": "7", "cy": "7", "r": "3" }]
  ];
  Icon($$renderer, spread_props([
    { name: "settings-2" },
    $$sanitized_props,
    {
      /**
       * @component @name Settings2
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMjAgN2gtOSIgLz4KICA8cGF0aCBkPSJNMTQgMTdINSIgLz4KICA8Y2lyY2xlIGN4PSIxNyIgY3k9IjE3IiByPSIzIiAvPgogIDxjaXJjbGUgY3g9IjciIGN5PSI3IiByPSIzIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/settings-2
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
function Table($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M12 3v18" }],
    [
      "rect",
      { "width": "18", "height": "18", "x": "3", "y": "3", "rx": "2" }
    ],
    ["path", { "d": "M3 9h18" }],
    ["path", { "d": "M3 15h18" }]
  ];
  Icon($$renderer, spread_props([
    { name: "table" },
    $$sanitized_props,
    {
      /**
       * @component @name Table
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTIgM3YxOCIgLz4KICA8cmVjdCB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHg9IjMiIHk9IjMiIHJ4PSIyIiAvPgogIDxwYXRoIGQ9Ik0zIDloMTgiIC8+CiAgPHBhdGggZD0iTTMgMTVoMTgiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/table
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
function Input($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { class: className, value = "", $$slots, $$events, ...rest } = $$props;
    $$renderer2.push(`<input${attributes(
      {
        value,
        class: clsx(cn("flex h-8 w-full rounded-md border border-border bg-background px-3 py-1 text-sm shadow-xs transition-colors", "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring", "disabled:cursor-not-allowed disabled:opacity-50", className)),
        ...rest
      },
      void 0,
      void 0,
      void 0,
      4
    )}/>`);
    bind_props($$props, { value });
  });
}
function box(initialValue) {
  let current = initialValue;
  return {
    [BoxSymbol]: true,
    [isWritableSymbol]: true,
    get current() {
      return current;
    },
    set current(v2) {
      current = v2;
    }
  };
}
box.from = boxFrom;
box.with = boxWith;
box.flatten = boxFlatten;
box.readonly = toReadonlyBox;
box.isBox = isBox;
box.isWritableBox = isWritableBox;
function getElemDirection(elem) {
  const style = window.getComputedStyle(elem);
  const direction = style.getPropertyValue("direction");
  return direction;
}
function getNextKey(dir = "ltr", orientation = "horizontal") {
  return {
    horizontal: dir === "rtl" ? ARROW_LEFT : ARROW_RIGHT,
    vertical: ARROW_DOWN
  }[orientation];
}
function getPrevKey(dir = "ltr", orientation = "horizontal") {
  return {
    horizontal: dir === "rtl" ? ARROW_RIGHT : ARROW_LEFT,
    vertical: ARROW_UP
  }[orientation];
}
function getDirectionalKeys(dir = "ltr", orientation = "horizontal") {
  if (!["ltr", "rtl"].includes(dir))
    dir = "ltr";
  if (!["horizontal", "vertical"].includes(orientation))
    orientation = "horizontal";
  return {
    nextKey: getNextKey(dir, orientation),
    prevKey: getPrevKey(dir, orientation)
  };
}
class RovingFocusGroup {
  #opts;
  #currentTabStopId = box(null);
  constructor(opts) {
    this.#opts = opts;
  }
  getCandidateNodes() {
    return [];
  }
  focusFirstCandidate() {
    const items = this.getCandidateNodes();
    if (!items.length)
      return;
    items[0]?.focus();
  }
  handleKeydown(node, e, both = false) {
    const rootNode = this.#opts.rootNode.current;
    if (!rootNode || !node)
      return;
    const items = this.getCandidateNodes();
    if (!items.length)
      return;
    const currentIndex = items.indexOf(node);
    const dir = getElemDirection(rootNode);
    const { nextKey, prevKey } = getDirectionalKeys(dir, this.#opts.orientation.current);
    const loop = this.#opts.loop.current;
    const keyToIndex = {
      [nextKey]: currentIndex + 1,
      [prevKey]: currentIndex - 1,
      [HOME]: 0,
      [END]: items.length - 1
    };
    if (both) {
      const altNextKey = nextKey === ARROW_DOWN ? ARROW_RIGHT : ARROW_DOWN;
      const altPrevKey = prevKey === ARROW_UP ? ARROW_LEFT : ARROW_UP;
      keyToIndex[altNextKey] = currentIndex + 1;
      keyToIndex[altPrevKey] = currentIndex - 1;
    }
    let itemIndex = keyToIndex[e.key];
    if (itemIndex === void 0)
      return;
    e.preventDefault();
    if (itemIndex < 0 && loop) {
      itemIndex = items.length - 1;
    } else if (itemIndex === items.length && loop) {
      itemIndex = 0;
    }
    const itemToFocus = items[itemIndex];
    if (!itemToFocus)
      return;
    itemToFocus.focus();
    this.#currentTabStopId.current = itemToFocus.id;
    this.#opts.onCandidateFocus?.(itemToFocus);
    return itemToFocus;
  }
  getTabIndex(node) {
    const items = this.getCandidateNodes();
    const anyActive = this.#currentTabStopId.current !== null;
    if (node && !anyActive && items[0] === node) {
      this.#currentTabStopId.current = node.id;
      return 0;
    } else if (node?.id === this.#currentTabStopId.current) {
      return 0;
    }
    return -1;
  }
  setCurrentTabStopId(id) {
    this.#currentTabStopId.current = id;
  }
  focusCurrentTabStop() {
    const currentTabStopId = this.#currentTabStopId.current;
    if (!currentTabStopId)
      return;
    const currentTabStop = this.#opts.rootNode.current?.querySelector(`#${currentTabStopId}`);
    if (!currentTabStop || !isHTMLElement(currentTabStop))
      return;
    currentTabStop.focus();
  }
}
class SvelteResizeObserver {
  #node;
  #onResize;
  constructor(node, onResize) {
    this.#node = node;
    this.#onResize = onResize;
    this.handler = this.handler.bind(this);
  }
  handler() {
    let rAF = 0;
    const _node = this.#node();
    if (!_node) return;
    const resizeObserver = new ResizeObserver(() => {
      cancelAnimationFrame(rAF);
      rAF = window.requestAnimationFrame(this.#onResize);
    });
    resizeObserver.observe(_node);
    return () => {
      window.cancelAnimationFrame(rAF);
      resizeObserver.unobserve(_node);
    };
  }
}
const radioGroupAttrs = createBitsAttrs({ component: "radio-group", parts: ["root", "item"] });
const RadioGroupRootContext = new Context("RadioGroup.Root");
class RadioGroupRootState {
  static create(opts) {
    return RadioGroupRootContext.set(new RadioGroupRootState(opts));
  }
  opts;
  #hasValue = derived(() => this.opts.value.current !== "");
  get hasValue() {
    return this.#hasValue();
  }
  set hasValue($$value) {
    return this.#hasValue($$value);
  }
  rovingFocusGroup;
  attachment;
  constructor(opts) {
    this.opts = opts;
    this.attachment = attachRef(this.opts.ref);
    this.rovingFocusGroup = new RovingFocusGroup({
      rootNode: this.opts.ref,
      candidateAttr: radioGroupAttrs.item,
      loop: this.opts.loop,
      orientation: this.opts.orientation
    });
  }
  isChecked(value) {
    return this.opts.value.current === value;
  }
  setValue(value) {
    this.opts.value.current = value;
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    role: "radiogroup",
    "aria-required": boolToStr(this.opts.required.current),
    "aria-disabled": boolToStr(this.opts.disabled.current),
    "aria-readonly": this.opts.readonly.current ? "true" : void 0,
    "data-disabled": boolToEmptyStrOrUndef(this.opts.disabled.current),
    "data-readonly": boolToEmptyStrOrUndef(this.opts.readonly.current),
    "data-orientation": this.opts.orientation.current,
    [radioGroupAttrs.root]: "",
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class RadioGroupItemState {
  static create(opts) {
    return new RadioGroupItemState(opts, RadioGroupRootContext.get());
  }
  opts;
  root;
  attachment;
  #checked = derived(() => this.root.opts.value.current === this.opts.value.current);
  get checked() {
    return this.#checked();
  }
  set checked($$value) {
    return this.#checked($$value);
  }
  #isDisabled = derived(() => this.opts.disabled.current || this.root.opts.disabled.current);
  #isReadonly = derived(() => this.root.opts.readonly.current);
  #isChecked = derived(() => this.root.isChecked(this.opts.value.current));
  #tabIndex = -1;
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(this.opts.ref);
    if (this.opts.value.current === this.root.opts.value.current) {
      this.root.rovingFocusGroup.setCurrentTabStopId(this.opts.id.current);
      this.#tabIndex = 0;
    } else if (!this.root.opts.value.current) {
      this.#tabIndex = 0;
    }
    watch(
      [
        () => this.opts.value.current,
        () => this.root.opts.value.current
      ],
      () => {
        if (this.opts.value.current === this.root.opts.value.current) {
          this.root.rovingFocusGroup.setCurrentTabStopId(this.opts.id.current);
          this.#tabIndex = 0;
        }
      }
    );
    this.onclick = this.onclick.bind(this);
    this.onkeydown = this.onkeydown.bind(this);
    this.onfocus = this.onfocus.bind(this);
  }
  onclick(_) {
    if (this.opts.disabled.current || this.#isReadonly()) return;
    this.root.setValue(this.opts.value.current);
  }
  onfocus(_) {
    if (!this.root.hasValue || this.#isReadonly()) return;
    this.root.setValue(this.opts.value.current);
  }
  onkeydown(e) {
    if (this.#isDisabled()) return;
    if (e.key === SPACE) {
      e.preventDefault();
      if (!this.#isReadonly()) {
        this.root.setValue(this.opts.value.current);
      }
      return;
    }
    this.root.rovingFocusGroup.handleKeydown(this.opts.ref.current, e, true);
  }
  #snippetProps = derived(() => ({ checked: this.#isChecked() }));
  get snippetProps() {
    return this.#snippetProps();
  }
  set snippetProps($$value) {
    return this.#snippetProps($$value);
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    disabled: this.#isDisabled() ? true : void 0,
    "data-value": this.opts.value.current,
    "data-orientation": this.root.opts.orientation.current,
    "data-disabled": boolToEmptyStrOrUndef(this.#isDisabled()),
    "data-readonly": boolToEmptyStrOrUndef(this.#isReadonly()),
    "data-state": this.#isChecked() ? "checked" : "unchecked",
    "aria-checked": getAriaChecked(this.#isChecked()),
    [radioGroupAttrs.item]: "",
    type: "button",
    role: "radio",
    tabindex: this.#tabIndex,
    onkeydown: this.onkeydown,
    onfocus: this.onfocus,
    onclick: this.onclick,
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class RadioGroupInputState {
  static create() {
    return new RadioGroupInputState(RadioGroupRootContext.get());
  }
  root;
  #shouldRender = derived(() => this.root.opts.name.current !== void 0);
  get shouldRender() {
    return this.#shouldRender();
  }
  set shouldRender($$value) {
    return this.#shouldRender($$value);
  }
  constructor(root) {
    this.root = root;
    this.onfocus = this.onfocus.bind(this);
  }
  onfocus(_) {
    this.root.rovingFocusGroup.focusCurrentTabStop();
  }
  #props = derived(() => ({
    name: this.root.opts.name.current,
    value: this.root.opts.value.current,
    required: this.root.opts.required.current,
    disabled: this.root.opts.disabled.current,
    onfocus: this.onfocus
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
function Radio_group_input($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const inputState = RadioGroupInputState.create();
    if (inputState.shouldRender) {
      $$renderer2.push("<!--[0-->");
      Hidden_input($$renderer2, spread_props([inputState.props]));
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function Radio_group$1($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      disabled = false,
      children,
      child,
      value = "",
      ref = null,
      orientation = "vertical",
      loop = true,
      name = void 0,
      required = false,
      readonly = false,
      id = createId(uid),
      onValueChange = noop,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const rootState = RadioGroupRootState.create({
      orientation: boxWith(() => orientation),
      disabled: boxWith(() => disabled),
      loop: boxWith(() => loop),
      name: boxWith(() => name),
      required: boxWith(() => required),
      readonly: boxWith(() => readonly),
      id: boxWith(() => id),
      value: boxWith(() => value, (v2) => {
        if (v2 === value) return;
        value = v2;
        onValueChange?.(v2);
      }),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, rootState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps() });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]--> `);
    Radio_group_input($$renderer2);
    $$renderer2.push(`<!---->`);
    bind_props($$props, { value, ref });
  });
}
function Radio_group_item($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      id = createId(uid),
      children,
      child,
      value,
      disabled = false,
      ref = null,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const itemState = RadioGroupItemState.create({
      value: boxWith(() => value),
      disabled: boxWith(() => disabled ?? false),
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, itemState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps(), ...itemState.snippetProps });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<button${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2, itemState.snippetProps);
      $$renderer2.push(`<!----></button>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function getRangeStyles(direction, min, max) {
  const styles = {
    position: "absolute"
  };
  if (direction === "lr") {
    styles.left = `${min}%`;
    styles.right = `${max}%`;
  } else if (direction === "rl") {
    styles.right = `${min}%`;
    styles.left = `${max}%`;
  } else if (direction === "bt") {
    styles.bottom = `${min}%`;
    styles.top = `${max}%`;
  } else {
    styles.top = `${min}%`;
    styles.bottom = `${max}%`;
  }
  return styles;
}
function getThumbStyles(direction, thumbPos) {
  const styles = {
    position: "absolute"
  };
  if (direction === "lr") {
    styles.left = `${thumbPos}%`;
    styles.translate = "-50% 0";
  } else if (direction === "rl") {
    styles.right = `${thumbPos}%`;
    styles.translate = "50% 0";
  } else if (direction === "bt") {
    styles.bottom = `${thumbPos}%`;
    styles.translate = "0 50%";
  } else {
    styles.top = `${thumbPos}%`;
    styles.translate = "0 -50%";
  }
  return styles;
}
function getTickStyles(direction, tickPosition, offsetPercentage) {
  const style = {
    position: "absolute"
  };
  if (direction === "lr") {
    style.left = `${tickPosition}%`;
    style.translate = `${offsetPercentage}% 0`;
  } else if (direction === "rl") {
    style.right = `${tickPosition}%`;
    style.translate = `${-offsetPercentage}% 0`;
  } else if (direction === "bt") {
    style.bottom = `${tickPosition}%`;
    style.translate = `0 ${-offsetPercentage}%`;
  } else {
    style.top = `${tickPosition}%`;
    style.translate = `0 ${offsetPercentage}%`;
  }
  return style;
}
function getDecimalPlaces(num) {
  if (Math.floor(num) === num)
    return 0;
  const str = num.toString();
  if (str.indexOf(".") !== -1 && str.indexOf("e-") === -1) {
    return str.split(".")[1].length;
  } else if (str.indexOf("e-") !== -1) {
    const parts = str.split("e-");
    return parseInt(parts[1], 10);
  }
  return 0;
}
function roundToPrecision(num, precision) {
  const factor = Math.pow(10, precision);
  return Math.round(num * factor) / factor;
}
function normalizeSteps(step, min, max) {
  if (typeof step === "number") {
    const difference = max - min;
    let count = Math.ceil(difference / step);
    const precision = getDecimalPlaces(step);
    const factor = Math.pow(10, precision);
    const intDifference = Math.round(difference * factor);
    const intStep = Math.round(step * factor);
    if (intDifference % intStep === 0) {
      count++;
    }
    const steps = [];
    for (let i = 0; i < count; i++) {
      const value = min + i * step;
      const roundedValue = roundToPrecision(value, precision);
      steps.push(roundedValue);
    }
    return steps;
  }
  return [...new Set(step)].filter((value) => value >= min && value <= max).sort((a, b) => a - b);
}
function snapValueToCustomSteps(value, steps) {
  if (steps.length === 0)
    return value;
  let closest = steps[0];
  let minDistance = Math.abs(value - closest);
  for (const step of steps) {
    const distance = Math.abs(value - step);
    if (distance < minDistance) {
      minDistance = distance;
      closest = step;
    }
  }
  return closest;
}
function getAdjacentStepValue(currentValue, steps, direction) {
  const currentIndex = steps.indexOf(currentValue);
  if (currentIndex === -1) {
    return snapValueToCustomSteps(currentValue, steps);
  }
  if (direction === "next") {
    return currentIndex < steps.length - 1 ? steps[currentIndex + 1] : currentValue;
  } else {
    return currentIndex > 0 ? steps[currentIndex - 1] : currentValue;
  }
}
function linearScale(domain, range, clamp = true) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const slope = (r1 - r0) / (d1 - d0);
  return (x) => {
    const result = r0 + slope * (x - d0);
    if (!clamp)
      return result;
    if (result > Math.max(r0, r1))
      return Math.max(r0, r1);
    if (result < Math.min(r0, r1))
      return Math.min(r0, r1);
    return result;
  };
}
const sliderAttrs = createBitsAttrs({
  component: "slider",
  parts: [
    "root",
    "thumb",
    "range",
    "tick",
    "tick-label",
    "thumb-label"
  ]
});
const SliderRootContext = new Context("Slider.Root");
class SliderBaseRootState {
  opts;
  attachment;
  isActive = false;
  #layoutVersion = 0;
  #direction = derived(() => {
    if (this.opts.orientation.current === "horizontal") {
      return this.opts.dir.current === "rtl" ? "rl" : "lr";
    } else {
      return this.opts.dir.current === "rtl" ? "tb" : "bt";
    }
  });
  get direction() {
    return this.#direction();
  }
  set direction($$value) {
    return this.#direction($$value);
  }
  #normalizedSteps = derived(() => {
    return normalizeSteps(this.opts.step.current, this.opts.min.current, this.opts.max.current);
  });
  get normalizedSteps() {
    return this.#normalizedSteps();
  }
  set normalizedSteps($$value) {
    return this.#normalizedSteps($$value);
  }
  domContext;
  constructor(opts) {
    this.opts = opts;
    this.attachment = attachRef(opts.ref);
    this.domContext = new DOMContext(this.opts.ref);
    new SvelteResizeObserver(() => this.opts.ref.current, this.#handleLayoutChange);
  }
  #handleLayoutChange = () => {
    this.#layoutVersion += 1;
  };
  isThumbActive(_index) {
    return this.isActive;
  }
  #touchAction = derived(() => {
    if (this.opts.disabled.current) return void 0;
    return this.opts.orientation.current === "horizontal" ? "pan-y" : "pan-x";
  });
  getAllThumbs = () => {
    const node = this.opts.ref.current;
    if (!node) return [];
    return Array.from(node.querySelectorAll(sliderAttrs.selector("thumb")));
  };
  getThumbScale = () => {
    void this.#layoutVersion;
    const trackPadding = this.opts.trackPadding?.current;
    if (trackPadding !== void 0 && trackPadding > 0) {
      return [trackPadding, 100 - trackPadding];
    }
    if (this.opts.thumbPositioning.current === "exact") {
      return [0, 100];
    }
    const isVertical = this.opts.orientation.current === "vertical";
    const activeThumb = this.getAllThumbs()[0];
    const thumbSize = isVertical ? activeThumb?.offsetHeight : activeThumb?.offsetWidth;
    if (thumbSize === void 0 || Number.isNaN(thumbSize) || thumbSize === 0) return [0, 100];
    const trackSize = isVertical ? this.opts.ref.current?.offsetHeight : this.opts.ref.current?.offsetWidth;
    if (trackSize === void 0 || Number.isNaN(trackSize) || trackSize === 0) return [0, 100];
    const percentPadding = thumbSize / 2 / trackSize * 100;
    const min = percentPadding;
    const max = 100 - percentPadding;
    return [min, max];
  };
  getPositionFromValue = (thumbValue) => {
    const thumbScale = this.getThumbScale();
    const scale = linearScale([this.opts.min.current, this.opts.max.current], thumbScale);
    return scale(thumbValue);
  };
  #props = derived(() => ({
    id: this.opts.id.current,
    "data-orientation": this.opts.orientation.current,
    "data-disabled": boolToEmptyStrOrUndef(this.opts.disabled.current),
    style: { touchAction: this.#touchAction() },
    [sliderAttrs.root]: "",
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class SliderSingleRootState extends SliderBaseRootState {
  opts;
  isMulti = false;
  constructor(opts) {
    super(opts);
    this.opts = opts;
    watch(
      [
        () => this.opts.step.current,
        () => this.opts.min.current,
        () => this.opts.max.current,
        () => this.opts.value.current
      ],
      ([step, min, max, value]) => {
        const steps = normalizeSteps(step, min, max);
        const isValidValue = (v2) => {
          return steps.includes(v2);
        };
        const gcv = (v2) => {
          return snapValueToCustomSteps(v2, steps);
        };
        if (!isValidValue(value)) {
          this.opts.value.current = gcv(value);
        }
      }
    );
  }
  isTickValueSelected = (tickValue) => {
    return this.opts.value.current === tickValue;
  };
  applyPosition({ clientXY, start, end }) {
    const min = this.opts.min.current;
    const max = this.opts.max.current;
    const percent = (clientXY - start) / (end - start);
    const val = percent * (max - min) + min;
    if (val < min) {
      this.updateValue(min);
    } else if (val > max) {
      this.updateValue(max);
    } else {
      const steps = this.normalizedSteps;
      const newValue = snapValueToCustomSteps(val, steps);
      this.updateValue(newValue);
    }
  }
  updateValue = (newValue) => {
    this.opts.value.current = snapValueToCustomSteps(newValue, this.normalizedSteps);
  };
  handlePointerMove = (e) => {
    if (!this.isActive || this.opts.disabled.current) return;
    e.preventDefault();
    e.stopPropagation();
    const sliderNode = this.opts.ref.current;
    const activeThumb = this.getAllThumbs()[0];
    if (!sliderNode || !activeThumb) return;
    activeThumb.focus();
    const { left, right, top, bottom } = sliderNode.getBoundingClientRect();
    if (this.direction === "lr") {
      this.applyPosition({ clientXY: e.clientX, start: left, end: right });
    } else if (this.direction === "rl") {
      this.applyPosition({ clientXY: e.clientX, start: right, end: left });
    } else if (this.direction === "bt") {
      this.applyPosition({ clientXY: e.clientY, start: bottom, end: top });
    } else if (this.direction === "tb") {
      this.applyPosition({ clientXY: e.clientY, start: top, end: bottom });
    }
  };
  handlePointerDown = (e) => {
    if (e.button !== 0 || this.opts.disabled.current) return;
    const sliderNode = this.opts.ref.current;
    const closestThumb = this.getAllThumbs()[0];
    if (!closestThumb || !sliderNode) return;
    const target = e.composedPath()[0] ?? e.target;
    if (!isElementOrSVGElement(target) || !sliderNode.contains(target)) return;
    e.preventDefault();
    closestThumb.focus();
    this.isActive = true;
    this.handlePointerMove(e);
  };
  handlePointerUp = () => {
    if (this.opts.disabled.current) return;
    if (this.isActive) {
      this.opts.onValueCommit.current(run(() => this.opts.value.current));
    }
    this.isActive = false;
  };
  #thumbsPropsArr = derived(() => {
    const currValue = this.opts.value.current;
    return Array.from({ length: 1 }, () => {
      const thumbValue = currValue;
      const thumbPosition = this.getPositionFromValue(thumbValue);
      const style = getThumbStyles(this.direction, thumbPosition);
      return {
        role: "slider",
        "aria-valuemin": this.opts.min.current,
        "aria-valuemax": this.opts.max.current,
        "aria-valuenow": thumbValue,
        "aria-disabled": boolToStr(this.opts.disabled.current),
        "aria-orientation": this.opts.orientation.current,
        "data-value": thumbValue,
        "data-orientation": this.opts.orientation.current,
        style,
        [sliderAttrs.thumb]: ""
      };
    });
  });
  get thumbsPropsArr() {
    return this.#thumbsPropsArr();
  }
  set thumbsPropsArr($$value) {
    return this.#thumbsPropsArr($$value);
  }
  #thumbsRenderArr = derived(() => {
    return this.thumbsPropsArr.map((_, i) => i);
  });
  get thumbsRenderArr() {
    return this.#thumbsRenderArr();
  }
  set thumbsRenderArr($$value) {
    return this.#thumbsRenderArr($$value);
  }
  #ticksPropsArr = derived(() => {
    const steps = this.normalizedSteps;
    const currValue = this.opts.value.current;
    return steps.map((tickValue, i) => {
      const tickPosition = this.getPositionFromValue(tickValue);
      const isFirst = i === 0;
      const isLast = i === steps.length - 1;
      const offsetPercentage = isFirst ? 0 : isLast ? -100 : -50;
      const style = getTickStyles(this.direction, tickPosition, offsetPercentage);
      const bounded = tickValue <= currValue;
      return {
        "data-disabled": boolToEmptyStrOrUndef(this.opts.disabled.current),
        "data-orientation": this.opts.orientation.current,
        "data-bounded": bounded ? "" : void 0,
        "data-value": tickValue,
        "data-selected": this.isTickValueSelected(tickValue) ? "" : void 0,
        style,
        [sliderAttrs.tick]: ""
      };
    });
  });
  get ticksPropsArr() {
    return this.#ticksPropsArr();
  }
  set ticksPropsArr($$value) {
    return this.#ticksPropsArr($$value);
  }
  #ticksRenderArr = derived(() => {
    return this.ticksPropsArr.map((_, i) => i);
  });
  get ticksRenderArr() {
    return this.#ticksRenderArr();
  }
  set ticksRenderArr($$value) {
    return this.#ticksRenderArr($$value);
  }
  #tickItemsArr = derived(() => {
    return this.ticksPropsArr.map((tick, i) => ({ value: tick["data-value"], index: i }));
  });
  get tickItemsArr() {
    return this.#tickItemsArr();
  }
  set tickItemsArr($$value) {
    return this.#tickItemsArr($$value);
  }
  #thumbItemsArr = derived(() => {
    const currValue = this.opts.value.current;
    return [{ value: currValue, index: 0 }];
  });
  get thumbItemsArr() {
    return this.#thumbItemsArr();
  }
  set thumbItemsArr($$value) {
    return this.#thumbItemsArr($$value);
  }
  #snippetProps = derived(() => ({
    ticks: this.ticksRenderArr,
    thumbs: this.thumbsRenderArr,
    tickItems: this.tickItemsArr,
    thumbItems: this.thumbItemsArr
  }));
  get snippetProps() {
    return this.#snippetProps();
  }
  set snippetProps($$value) {
    return this.#snippetProps($$value);
  }
}
class SliderMultiRootState extends SliderBaseRootState {
  opts;
  isMulti = true;
  activeThumb = null;
  currentThumbIdx = 0;
  constructor(opts) {
    super(opts);
    this.opts = opts;
    watch(
      [
        () => this.opts.step.current,
        () => this.opts.min.current,
        () => this.opts.max.current,
        () => this.opts.value.current
      ],
      ([step, min, max, value]) => {
        const steps = normalizeSteps(step, min, max);
        const isValidValue = (v2) => {
          return steps.includes(v2);
        };
        const gcv = (v2) => {
          return snapValueToCustomSteps(v2, steps);
        };
        if (value.some((v2) => !isValidValue(v2))) {
          this.opts.value.current = value.map(gcv);
        }
      }
    );
  }
  isTickValueSelected = (tickValue) => {
    return this.opts.value.current.includes(tickValue);
  };
  isThumbActive(index) {
    return this.isActive && this.activeThumb?.idx === index;
  }
  applyPosition({ clientXY, activeThumbIdx, start, end }) {
    const min = this.opts.min.current;
    const max = this.opts.max.current;
    const percent = (clientXY - start) / (end - start);
    const val = percent * (max - min) + min;
    if (val < min) {
      this.updateValue(min, activeThumbIdx);
    } else if (val > max) {
      this.updateValue(max, activeThumbIdx);
    } else {
      const steps = this.normalizedSteps;
      const newValue = snapValueToCustomSteps(val, steps);
      this.updateValue(newValue, activeThumbIdx);
    }
  }
  #getClosestThumb = (e) => {
    const thumbs = this.getAllThumbs();
    if (!thumbs.length) return;
    for (const thumb of thumbs) {
      thumb.blur();
    }
    const distances = thumbs.map((thumb) => {
      if (this.opts.orientation.current === "horizontal") {
        const { left, right } = thumb.getBoundingClientRect();
        return Math.abs(e.clientX - (left + right) / 2);
      } else {
        const { top, bottom } = thumb.getBoundingClientRect();
        return Math.abs(e.clientY - (top + bottom) / 2);
      }
    });
    const node = thumbs[distances.indexOf(Math.min(...distances))];
    const idx = thumbs.indexOf(node);
    return { node, idx };
  };
  handlePointerMove = (e) => {
    if (!this.isActive || this.opts.disabled.current) return;
    e.preventDefault();
    e.stopPropagation();
    const sliderNode = this.opts.ref.current;
    const activeThumb = this.activeThumb;
    if (!sliderNode || !activeThumb) return;
    activeThumb.node.focus();
    const { left, right, top, bottom } = sliderNode.getBoundingClientRect();
    const direction = this.direction;
    if (direction === "lr") {
      this.applyPosition({
        clientXY: e.clientX,
        activeThumbIdx: activeThumb.idx,
        start: left,
        end: right
      });
    } else if (direction === "rl") {
      this.applyPosition({
        clientXY: e.clientX,
        activeThumbIdx: activeThumb.idx,
        start: right,
        end: left
      });
    } else if (direction === "bt") {
      this.applyPosition({
        clientXY: e.clientY,
        activeThumbIdx: activeThumb.idx,
        start: bottom,
        end: top
      });
    } else if (direction === "tb") {
      this.applyPosition({
        clientXY: e.clientY,
        activeThumbIdx: activeThumb.idx,
        start: top,
        end: bottom
      });
    }
  };
  handlePointerDown = (e) => {
    if (e.button !== 0 || this.opts.disabled.current) return;
    const sliderNode = this.opts.ref.current;
    const closestThumb = this.#getClosestThumb(e);
    if (!closestThumb || !sliderNode) return;
    const target = e.composedPath()[0] ?? e.target;
    if (!isElementOrSVGElement(target) || !sliderNode.contains(target)) return;
    e.preventDefault();
    this.activeThumb = closestThumb;
    closestThumb.node.focus();
    this.isActive = true;
    this.handlePointerMove(e);
  };
  handlePointerUp = () => {
    if (this.opts.disabled.current) return;
    if (this.isActive) {
      this.opts.onValueCommit.current(run(() => this.opts.value.current));
    }
    this.isActive = false;
  };
  getAllThumbs = () => {
    const node = this.opts.ref.current;
    if (!node) return [];
    const thumbs = Array.from(node.querySelectorAll(sliderAttrs.selector("thumb")));
    return thumbs;
  };
  updateValue = (thumbValue, idx) => {
    const currValue = this.opts.value.current;
    if (!currValue.length) {
      this.opts.value.current.push(thumbValue);
      return;
    }
    const valueAtIndex = currValue[idx];
    if (valueAtIndex === thumbValue) return;
    const newValue = [...currValue];
    if (!isValidIndex(idx, newValue)) return;
    const direction = newValue[idx] > thumbValue ? -1 : 1;
    const swap = () => {
      const diffIndex = idx + direction;
      newValue[idx] = newValue[diffIndex];
      newValue[diffIndex] = thumbValue;
      const thumbs = this.getAllThumbs();
      if (!thumbs.length) return;
      thumbs[diffIndex]?.focus();
      this.activeThumb = { node: thumbs[diffIndex], idx: diffIndex };
    };
    if (this.opts.autoSort.current && (direction === -1 && thumbValue < newValue[idx - 1] || direction === 1 && thumbValue > newValue[idx + 1])) {
      swap();
      this.opts.value.current = newValue;
      return;
    }
    const steps = this.normalizedSteps;
    newValue[idx] = snapValueToCustomSteps(thumbValue, steps);
    this.opts.value.current = newValue;
  };
  #thumbsPropsArr = derived(() => {
    const currValue = this.opts.value.current;
    return Array.from({ length: currValue.length || 1 }, (_, i) => {
      const currThumb = run(() => this.currentThumbIdx);
      if (currThumb < currValue.length) {
        run(() => {
          this.currentThumbIdx = currThumb + 1;
        });
      }
      const thumbValue = currValue[i];
      const thumbPosition = this.getPositionFromValue(thumbValue ?? 0);
      const style = getThumbStyles(this.direction, thumbPosition);
      return {
        role: "slider",
        "aria-valuemin": this.opts.min.current,
        "aria-valuemax": this.opts.max.current,
        "aria-valuenow": thumbValue,
        "aria-disabled": boolToStr(this.opts.disabled.current),
        "aria-orientation": this.opts.orientation.current,
        "data-value": thumbValue,
        "data-orientation": this.opts.orientation.current,
        style,
        [sliderAttrs.thumb]: ""
      };
    });
  });
  get thumbsPropsArr() {
    return this.#thumbsPropsArr();
  }
  set thumbsPropsArr($$value) {
    return this.#thumbsPropsArr($$value);
  }
  #thumbsRenderArr = derived(() => {
    return this.thumbsPropsArr.map((_, i) => i);
  });
  get thumbsRenderArr() {
    return this.#thumbsRenderArr();
  }
  set thumbsRenderArr($$value) {
    return this.#thumbsRenderArr($$value);
  }
  #ticksPropsArr = derived(() => {
    const steps = this.normalizedSteps;
    const currValue = this.opts.value.current;
    return steps.map((tickValue, i) => {
      const tickPosition = this.getPositionFromValue(tickValue);
      const isFirst = i === 0;
      const isLast = i === steps.length - 1;
      const offsetPercentage = isFirst ? 0 : isLast ? -100 : -50;
      const style = getTickStyles(this.direction, tickPosition, offsetPercentage);
      const bounded = currValue.length === 1 ? tickValue <= currValue[0] : currValue[0] <= tickValue && tickValue <= currValue[currValue.length - 1];
      return {
        "data-disabled": boolToEmptyStrOrUndef(this.opts.disabled.current),
        "data-orientation": this.opts.orientation.current,
        "data-bounded": bounded ? "" : void 0,
        "data-value": tickValue,
        style,
        [sliderAttrs.tick]: ""
      };
    });
  });
  get ticksPropsArr() {
    return this.#ticksPropsArr();
  }
  set ticksPropsArr($$value) {
    return this.#ticksPropsArr($$value);
  }
  #ticksRenderArr = derived(() => {
    return this.ticksPropsArr.map((_, i) => i);
  });
  get ticksRenderArr() {
    return this.#ticksRenderArr();
  }
  set ticksRenderArr($$value) {
    return this.#ticksRenderArr($$value);
  }
  #tickItemsArr = derived(() => {
    return this.ticksPropsArr.map((tick, i) => ({ value: tick["data-value"], index: i }));
  });
  get tickItemsArr() {
    return this.#tickItemsArr();
  }
  set tickItemsArr($$value) {
    return this.#tickItemsArr($$value);
  }
  #thumbItemsArr = derived(() => {
    const currValue = this.opts.value.current;
    return currValue.map((value, index) => ({ value, index }));
  });
  get thumbItemsArr() {
    return this.#thumbItemsArr();
  }
  set thumbItemsArr($$value) {
    return this.#thumbItemsArr($$value);
  }
  #snippetProps = derived(() => ({
    ticks: this.ticksRenderArr,
    thumbs: this.thumbsRenderArr,
    tickItems: this.tickItemsArr,
    thumbItems: this.thumbItemsArr
  }));
  get snippetProps() {
    return this.#snippetProps();
  }
  set snippetProps($$value) {
    return this.#snippetProps($$value);
  }
}
class SliderRootState {
  static create(opts) {
    const { type, ...rest } = opts;
    const rootState = type === "single" ? new SliderSingleRootState(rest) : new SliderMultiRootState(rest);
    return SliderRootContext.set(rootState);
  }
}
const VALID_SLIDER_KEYS = [
  ARROW_LEFT,
  ARROW_RIGHT,
  ARROW_UP,
  ARROW_DOWN,
  HOME,
  END
];
class SliderRangeState {
  static create(opts) {
    return new SliderRangeState(opts, SliderRootContext.get());
  }
  opts;
  root;
  attachment;
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(opts.ref);
  }
  #rangeStyles = derived(() => {
    if (Array.isArray(this.root.opts.value.current)) {
      const min = this.root.opts.value.current.length > 1 ? this.root.getPositionFromValue(Math.min(...this.root.opts.value.current) ?? 0) : 0;
      const max = 100 - this.root.getPositionFromValue(Math.max(...this.root.opts.value.current) ?? 0);
      return {
        position: "absolute",
        ...getRangeStyles(this.root.direction, min, max)
      };
    } else {
      const trackPadding = this.root.opts.trackPadding?.current;
      const currentValue = this.root.opts.value.current;
      const maxValue = this.root.opts.max.current;
      const min = 0;
      const max = trackPadding !== void 0 && trackPadding > 0 && currentValue === maxValue ? 0 : (
        // 100% - 0% = full width
        100 - this.root.getPositionFromValue(currentValue)
      );
      return {
        position: "absolute",
        ...getRangeStyles(this.root.direction, min, max)
      };
    }
  });
  get rangeStyles() {
    return this.#rangeStyles();
  }
  set rangeStyles($$value) {
    return this.#rangeStyles($$value);
  }
  #props = derived(() => ({
    id: this.opts.id.current,
    "data-orientation": this.root.opts.orientation.current,
    "data-disabled": boolToEmptyStrOrUndef(this.root.opts.disabled.current),
    style: this.rangeStyles,
    [sliderAttrs.range]: "",
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class SliderThumbState {
  static create(opts) {
    return new SliderThumbState(opts, SliderRootContext.get());
  }
  opts;
  root;
  attachment;
  #isDisabled = derived(() => this.root.opts.disabled.current || this.opts.disabled.current);
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(opts.ref);
    this.onkeydown = this.onkeydown.bind(this);
  }
  #updateValue(newValue) {
    if (this.root.isMulti) {
      this.root.updateValue(newValue, this.opts.index.current);
    } else {
      this.root.updateValue(newValue);
    }
  }
  onkeydown(e) {
    if (this.#isDisabled()) return;
    const currNode = this.opts.ref.current;
    if (!currNode) return;
    const thumbs = this.root.getAllThumbs();
    if (!thumbs.length) return;
    const idx = thumbs.indexOf(currNode);
    if (this.root.isMulti) {
      this.root.currentThumbIdx = idx;
    }
    if (!VALID_SLIDER_KEYS.includes(e.key)) return;
    e.preventDefault();
    const min = this.root.opts.min.current;
    const max = this.root.opts.max.current;
    const value = this.root.opts.value.current;
    const thumbValue = Array.isArray(value) ? value[idx] : value;
    const orientation = this.root.opts.orientation.current;
    const direction = this.root.direction;
    const steps = this.root.normalizedSteps;
    switch (e.key) {
      case HOME:
        this.#updateValue(min);
        break;
      case END:
        this.#updateValue(max);
        break;
      case ARROW_LEFT:
        if (orientation !== "horizontal") break;
        if (e.metaKey) {
          const newValue = direction === "rl" ? max : min;
          this.#updateValue(newValue);
        } else {
          const stepDirection = direction === "rl" ? "next" : "prev";
          const newValue = getAdjacentStepValue(thumbValue, steps, stepDirection);
          this.#updateValue(newValue);
        }
        break;
      case ARROW_RIGHT:
        if (orientation !== "horizontal") break;
        if (e.metaKey) {
          const newValue = direction === "rl" ? min : max;
          this.#updateValue(newValue);
        } else {
          const stepDirection = direction === "rl" ? "prev" : "next";
          const newValue = getAdjacentStepValue(thumbValue, steps, stepDirection);
          this.#updateValue(newValue);
        }
        break;
      case ARROW_UP:
        if (e.metaKey) {
          const newValue = direction === "tb" ? min : max;
          this.#updateValue(newValue);
        } else {
          const stepDirection = direction === "tb" ? "prev" : "next";
          const newValue = getAdjacentStepValue(thumbValue, steps, stepDirection);
          this.#updateValue(newValue);
        }
        break;
      case ARROW_DOWN:
        if (e.metaKey) {
          const newValue = direction === "tb" ? max : min;
          this.#updateValue(newValue);
        } else {
          const stepDirection = direction === "tb" ? "next" : "prev";
          const newValue = getAdjacentStepValue(thumbValue, steps, stepDirection);
          this.#updateValue(newValue);
        }
        break;
    }
    this.root.opts.onValueCommit.current(this.root.opts.value.current);
  }
  #props = derived(() => ({
    ...this.root.thumbsPropsArr[this.opts.index.current],
    id: this.opts.id.current,
    onkeydown: this.onkeydown,
    "data-active": this.root.isThumbActive(this.opts.index.current) ? "" : void 0,
    "data-disabled": boolToEmptyStrOrUndef(this.opts.disabled.current || this.root.opts.disabled.current),
    tabindex: this.opts.disabled.current || this.root.opts.disabled.current ? -1 : 0,
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
function Slider($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      children,
      child,
      id = createId(uid),
      ref = null,
      value = void 0,
      type,
      onValueChange = noop,
      onValueCommit = noop,
      disabled = false,
      min: minProp,
      max: maxProp,
      step = 1,
      dir = "ltr",
      autoSort = true,
      orientation = "horizontal",
      thumbPositioning = "contain",
      trackPadding,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const min = derived(() => {
      if (minProp !== void 0) return minProp;
      if (Array.isArray(step)) return Math.min(...step);
      return 0;
    });
    const max = derived(() => {
      if (maxProp !== void 0) return maxProp;
      if (Array.isArray(step)) return Math.max(...step);
      return 100;
    });
    function handleDefaultValue() {
      if (value !== void 0) return;
      if (type === "single") {
        return min();
      }
      return [];
    }
    handleDefaultValue();
    watch.pre(() => value, () => {
      handleDefaultValue();
    });
    const rootState = SliderRootState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2),
      value: boxWith(() => value, (v2) => {
        value = v2;
        onValueChange(v2);
      }),
      // @ts-expect-error - we know
      onValueCommit: boxWith(() => onValueCommit),
      disabled: boxWith(() => disabled),
      min: boxWith(() => min()),
      max: boxWith(() => max()),
      step: boxWith(() => step),
      dir: boxWith(() => dir),
      autoSort: boxWith(() => autoSort),
      orientation: boxWith(() => orientation),
      thumbPositioning: boxWith(() => thumbPositioning),
      type,
      trackPadding: boxWith(() => trackPadding)
    });
    const mergedProps = derived(() => mergeProps(restProps, rootState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps(), ...rootState.snippetProps });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<span${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2, rootState.snippetProps);
      $$renderer2.push(`<!----></span>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref, value });
  });
}
function Slider_range($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      children,
      child,
      ref = null,
      id = createId(uid),
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const rangeState = SliderRangeState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, rangeState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps() });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<span${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></span>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function Slider_thumb($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      children,
      child,
      ref = null,
      id = createId(uid),
      index,
      disabled = false,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const thumbState = SliderThumbState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2),
      index: boxWith(() => index),
      disabled: boxWith(() => disabled)
    });
    const mergedProps = derived(() => mergeProps(restProps, thumbState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, {
        active: thumbState.root.isThumbActive(thumbState.opts.index.current),
        props: mergedProps()
      });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<span${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2, {
        active: thumbState.root.isThumbActive(thumbState.opts.index.current)
      });
      $$renderer2.push(`<!----></span>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
const switchAttrs = createBitsAttrs({ component: "switch", parts: ["root", "thumb"] });
const SwitchRootContext = new Context("Switch.Root");
class SwitchRootState {
  static create(opts) {
    return SwitchRootContext.set(new SwitchRootState(opts));
  }
  opts;
  attachment;
  constructor(opts) {
    this.opts = opts;
    this.attachment = attachRef(opts.ref);
    this.onkeydown = this.onkeydown.bind(this);
    this.onclick = this.onclick.bind(this);
  }
  #toggle() {
    this.opts.checked.current = !this.opts.checked.current;
  }
  onkeydown(e) {
    if (!(e.key === ENTER || e.key === SPACE) || this.opts.disabled.current) return;
    e.preventDefault();
    this.#toggle();
  }
  onclick(_) {
    if (this.opts.disabled.current) return;
    this.#toggle();
  }
  #sharedProps = derived(() => ({
    "data-disabled": boolToEmptyStrOrUndef(this.opts.disabled.current),
    "data-state": getDataChecked(this.opts.checked.current),
    "data-required": boolToEmptyStrOrUndef(this.opts.required.current)
  }));
  get sharedProps() {
    return this.#sharedProps();
  }
  set sharedProps($$value) {
    return this.#sharedProps($$value);
  }
  #snippetProps = derived(() => ({ checked: this.opts.checked.current }));
  get snippetProps() {
    return this.#snippetProps();
  }
  set snippetProps($$value) {
    return this.#snippetProps($$value);
  }
  #props = derived(() => ({
    ...this.sharedProps,
    id: this.opts.id.current,
    role: "switch",
    disabled: boolToTrueOrUndef(this.opts.disabled.current),
    "aria-checked": getAriaChecked(this.opts.checked.current),
    "aria-required": boolToStr(this.opts.required.current),
    [switchAttrs.root]: "",
    onclick: this.onclick,
    onkeydown: this.onkeydown,
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class SwitchInputState {
  static create() {
    return new SwitchInputState(SwitchRootContext.get());
  }
  root;
  #shouldRender = derived(() => this.root.opts.name.current !== void 0);
  get shouldRender() {
    return this.#shouldRender();
  }
  set shouldRender($$value) {
    return this.#shouldRender($$value);
  }
  constructor(root) {
    this.root = root;
  }
  #props = derived(() => ({
    type: "checkbox",
    name: this.root.opts.name.current,
    value: this.root.opts.value.current,
    checked: this.root.opts.checked.current,
    disabled: this.root.opts.disabled.current,
    required: this.root.opts.required.current
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
class SwitchThumbState {
  static create(opts) {
    return new SwitchThumbState(opts, SwitchRootContext.get());
  }
  opts;
  root;
  attachment;
  constructor(opts, root) {
    this.opts = opts;
    this.root = root;
    this.attachment = attachRef(opts.ref);
  }
  #snippetProps = derived(() => ({ checked: this.root.opts.checked.current }));
  get snippetProps() {
    return this.#snippetProps();
  }
  set snippetProps($$value) {
    return this.#snippetProps($$value);
  }
  #props = derived(() => ({
    ...this.root.sharedProps,
    id: this.opts.id.current,
    [switchAttrs.thumb]: "",
    ...this.attachment
  }));
  get props() {
    return this.#props();
  }
  set props($$value) {
    return this.#props($$value);
  }
}
function Switch_input($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const inputState = SwitchInputState.create();
    if (inputState.shouldRender) {
      $$renderer2.push("<!--[0-->");
      Hidden_input($$renderer2, spread_props([inputState.props]));
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function Switch($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      child,
      children,
      ref = null,
      id = createId(uid),
      disabled = false,
      required = false,
      checked = false,
      value = "on",
      name = void 0,
      type = "button",
      onCheckedChange = noop,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const rootState = SwitchRootState.create({
      checked: boxWith(() => checked, (v2) => {
        checked = v2;
        onCheckedChange?.(v2);
      }),
      disabled: boxWith(() => disabled ?? false),
      required: boxWith(() => required),
      value: boxWith(() => value),
      name: boxWith(() => name),
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, rootState.props, { type }));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps(), ...rootState.snippetProps });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<button${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2, rootState.snippetProps);
      $$renderer2.push(`<!----></button>`);
    }
    $$renderer2.push(`<!--]--> `);
    Switch_input($$renderer2);
    $$renderer2.push(`<!---->`);
    bind_props($$props, { ref, checked });
  });
}
function Switch_thumb($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const uid = props_id($$renderer2);
    let {
      child,
      children,
      ref = null,
      id = createId(uid),
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const thumbState = SwitchThumbState.create({
      id: boxWith(() => id),
      ref: boxWith(() => ref, (v2) => ref = v2)
    });
    const mergedProps = derived(() => mergeProps(restProps, thumbState.props));
    if (child) {
      $$renderer2.push("<!--[0-->");
      child($$renderer2, { props: mergedProps(), ...thumbState.snippetProps });
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<span${attributes({ ...mergedProps() })}>`);
      children?.($$renderer2, thumbState.snippetProps);
      $$renderer2.push(`<!----></span>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function Field($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      label,
      description,
      inline = false,
      class: className,
      children
    } = $$props;
    $$renderer2.push(`<div${attr_class(clsx(cn(
      inline ? "flex items-center justify-between gap-3" : "flex flex-col gap-1.5",
      className
    )))}>`);
    if (label) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<span class="text-xs font-medium text-foreground">${escape_html(label)}</span>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--> <div${attr_class(clsx(inline ? "flex items-center gap-2" : "contents"))}>`);
    children($$renderer2);
    $$renderer2.push(`<!----></div> `);
    if (description && !inline) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<span class="text-[11px] text-muted-foreground">${escape_html(description)}</span>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function Switch_1($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      checked = false,
      class: className,
      $$slots,
      $$events,
      ...rest
    } = $$props;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (Switch) {
        $$renderer3.push("<!--[-->");
        Switch($$renderer3, spread_props([
          {
            "data-slot": "switch",
            class: cn("inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border border-transparent transition-colors", "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring", "data-[state=checked]:bg-primary data-[state=unchecked]:bg-secondary", className)
          },
          rest,
          {
            get checked() {
              return checked;
            },
            set checked($$value) {
              checked = $$value;
              $$settled = false;
            },
            children: ($$renderer4) => {
              if (Switch_thumb) {
                $$renderer4.push("<!--[-->");
                Switch_thumb($$renderer4, {
                  class: "pointer-events-none block size-4 rounded-full bg-background shadow transition-transform data-[state=checked]:translate-x-4 data-[state=unchecked]:translate-x-0.5"
                });
                $$renderer4.push("<!--]-->");
              } else {
                $$renderer4.push("<!--[!-->");
                $$renderer4.push("<!--]-->");
              }
            },
            $$slots: { default: true }
          }
        ]));
        $$renderer3.push("<!--]-->");
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push("<!--]-->");
      }
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { checked });
  });
}
function Slider_1($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = 50,
      min = 0,
      max = 100,
      step = 1,
      class: className,
      $$slots,
      $$events,
      ...rest
    } = $$props;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      {
        let children = function($$renderer4, { thumbs }) {
          $$renderer4.push(`<span class="relative h-1.5 w-full grow overflow-hidden rounded-full bg-secondary">`);
          if (Slider_range) {
            $$renderer4.push("<!--[-->");
            Slider_range($$renderer4, { class: "absolute h-full bg-primary" });
            $$renderer4.push("<!--]-->");
          } else {
            $$renderer4.push("<!--[!-->");
            $$renderer4.push("<!--]-->");
          }
          $$renderer4.push(`</span> <!--[-->`);
          const each_array = ensure_array_like(thumbs);
          for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
            let index = each_array[$$index];
            if (Slider_thumb) {
              $$renderer4.push("<!--[-->");
              Slider_thumb($$renderer4, {
                index,
                class: "block size-4 rounded-full border-2 border-primary bg-background shadow transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              });
              $$renderer4.push("<!--]-->");
            } else {
              $$renderer4.push("<!--[!-->");
              $$renderer4.push("<!--]-->");
            }
          }
          $$renderer4.push(`<!--]-->`);
        };
        if (Slider) {
          $$renderer3.push("<!--[-->");
          Slider($$renderer3, spread_props([
            {
              type: "single",
              min,
              max,
              step,
              "data-slot": "slider",
              class: cn("relative flex w-full touch-none items-center select-none", className)
            },
            rest,
            {
              get value() {
                return value;
              },
              set value($$value) {
                value = $$value;
                $$settled = false;
              },
              children,
              $$slots: { default: true }
            }
          ]));
          $$renderer3.push("<!--]-->");
        } else {
          $$renderer3.push("<!--[!-->");
          $$renderer3.push("<!--]-->");
        }
      }
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { value });
  });
}
function Radio_group($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { value = "", options, class: className } = $$props;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (Radio_group$1) {
        $$renderer3.push("<!--[-->");
        Radio_group$1($$renderer3, {
          "data-slot": "radio-group",
          class: cn("grid gap-1", className),
          get value() {
            return value;
          },
          set value($$value) {
            value = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            $$renderer4.push(`<!--[-->`);
            const each_array = ensure_array_like(options);
            for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
              let opt = each_array[$$index];
              $$renderer4.push(`<label class="flex cursor-pointer items-start gap-2 rounded p-1.5 transition-colors hover:bg-secondary/40">`);
              {
                let children = function($$renderer5, { checked }) {
                  if (checked) {
                    $$renderer5.push("<!--[0-->");
                    $$renderer5.push(`<span class="size-2 rounded-full bg-primary"></span>`);
                  } else {
                    $$renderer5.push("<!--[-1-->");
                  }
                  $$renderer5.push(`<!--]-->`);
                };
                if (Radio_group_item) {
                  $$renderer4.push("<!--[-->");
                  Radio_group_item($$renderer4, {
                    value: opt.value,
                    class: "mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full border border-border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring data-[state=checked]:border-primary",
                    children,
                    $$slots: { default: true }
                  });
                  $$renderer4.push("<!--]-->");
                } else {
                  $$renderer4.push("<!--[!-->");
                  $$renderer4.push("<!--]-->");
                }
              }
              $$renderer4.push(` <span class="flex-1 text-xs"><span class="font-medium text-foreground">${escape_html(opt.label)}</span> `);
              if (opt.description) {
                $$renderer4.push("<!--[0-->");
                $$renderer4.push(`<span class="block text-muted-foreground">${escape_html(opt.description)}</span>`);
              } else {
                $$renderer4.push("<!--[-1-->");
              }
              $$renderer4.push(`<!--]--></span></label>`);
            }
            $$renderer4.push(`<!--]-->`);
          },
          $$slots: { default: true }
        });
        $$renderer3.push("<!--]-->");
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push("<!--]-->");
      }
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { value });
  });
}
class VoiceSearchStore {
  /** `null` until the one-shot status probe resolves. */
  status = null;
  /** Latest unconsumed voice-search request (the search page consumes it). */
  pending = null;
  /** Settings toggle: also surface hits from the anchor's own video. */
  includeSameDoc = false;
  /** In-flight uploaded-clip fetches; a counter (not a boolean) so a rapid
   *  second pick can't clear the spinner while its successor is still
   *  running. The first upload after a backend restart lazy-loads the voice
   *  encoder (~5 s), so the attach button must visibly spin. */
  #uploadsInFlight = 0;
  #probed = false;
  /** True once the backend reports the voice tables are built. */
  get built() {
    return this.status?.built ?? false;
  }
  /** Fetch `/api/voice/status` once; later calls are no-ops. A failed probe
   *  reads as "not built" (all voice UI stays hidden) — never throws. */
  probe() {
    if (this.#probed) return;
    this.#probed = true;
    getVoiceStatus().then((s) => {
      this.status = s;
    }).catch(() => {
      this.status = { built: false, turns: 0, speakers: 0 };
    });
  }
  /** True while an uploaded-clip search is in flight (attach-button spinner). */
  get uploadPending() {
    return this.#uploadsInFlight > 0;
  }
  /** Start a voice search for `anchor` (auto-applies — no submit step). */
  request(anchor, label, fallback) {
    this.pending = { anchor, label, fallback };
  }
  /** Start a voice search ranked against an uploaded audio/video clip
   *  (auto-applies, same pending/chip lifecycle as `request`).
   *  `exclude_same_doc` doesn't apply — the clip isn't a Lance row. */
  requestUpload(file) {
    this.pending = { upload: file, label: `uploaded clip: ${file.name}` };
  }
  /** Bracket one uploaded-clip fetch (the search page wraps its
   *  `voiceSimilarUpload` call) so `uploadPending` tracks it. */
  beginUpload() {
    this.#uploadsInFlight += 1;
  }
  endUpload() {
    this.#uploadsInFlight = Math.max(0, this.#uploadsInFlight - 1);
  }
}
const voiceSearch = new VoiceSearchStore();
function Filter_popover($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { spec = void 0, onchange } = $$props;
    let whereSql = spec.where ?? "";
    function commit() {
      spec = { ...spec, where: whereSql || void 0 };
      onchange?.();
    }
    const activeCount = derived(() => whereSql.trim() ? 1 : 0);
    const wherePlaceholder = derived(() => {
      const fields = activeView().filterFields;
      return fields.length > 0 ? `${fields[0]} LIKE '%…%'` : "duration > 60";
    });
    let columns = [];
    let hidden = /* @__PURE__ */ new Set();
    const visibleCols = derived(() => columns.filter((c) => !hidden.has(c.name)));
    const NUMBER_OPS = [
      { value: "=", label: "=" },
      { value: "!=", label: "≠" },
      { value: ">", label: ">" },
      { value: ">=", label: "≥" },
      { value: "<", label: "<" },
      { value: "<=", label: "≤" }
    ];
    const TEXT_OPS = [
      { value: "contains", label: "contains" },
      { value: "equals", label: "equals" },
      { value: "starts", label: "starts with" }
    ];
    const BOOL_OPS = [
      { value: "=", label: "is" },
      { value: "!=", label: "is not" }
    ];
    let colName = "";
    let op = "contains";
    let val = "";
    const colType = derived(() => columns.find((c) => c.name === colName)?.type ?? "text");
    const opOptions = derived(() => colType() === "number" || colType() === "time" ? NUMBER_OPS : colType() === "boolean" ? BOOL_OPS : TEXT_OPS);
    function buildClause() {
      return null;
    }
    function addFilter() {
      const clause = buildClause();
      if (!clause) return;
      whereSql = whereSql.trim() ? `${whereSql.trim()} AND ${clause}` : clause;
      val = "";
      commit();
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (Popover) {
        $$renderer3.push("<!--[-->");
        Popover($$renderer3, {
          children: ($$renderer4) => {
            if (Popover_trigger) {
              $$renderer4.push("<!--[-->");
              Popover_trigger($$renderer4, {
                class: cn("inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-background px-3 text-xs transition-colors", activeCount() > 0 ? "text-foreground" : "text-muted-foreground", "hover:bg-muted hover:text-foreground data-[state=open]:bg-muted data-[state=open]:text-foreground"),
                title: "Filter results by any column",
                children: ($$renderer5) => {
                  Filter($$renderer5, { class: "size-3.5" });
                  $$renderer5.push(`<!----> <span>Filters</span> `);
                  if (activeCount() > 0) {
                    $$renderer5.push("<!--[0-->");
                    $$renderer5.push(`<span class="ml-1 rounded-full bg-primary px-1.5 text-[10px] font-bold text-primary-foreground">${escape_html(activeCount())}</span>`);
                  } else {
                    $$renderer5.push("<!--[-1-->");
                  }
                  $$renderer5.push(`<!--]-->`);
                },
                $$slots: { default: true }
              });
              $$renderer4.push("<!--]-->");
            } else {
              $$renderer4.push("<!--[!-->");
              $$renderer4.push("<!--]-->");
            }
            $$renderer4.push(` `);
            if (Portal) {
              $$renderer4.push("<!--[-->");
              Portal($$renderer4, {
                children: ($$renderer5) => {
                  if (Popover_content) {
                    $$renderer5.push("<!--[-->");
                    Popover_content($$renderer5, {
                      sideOffset: 6,
                      align: "end",
                      class: "z-50 flex max-h-[75vh] w-[340px] flex-col gap-3 overflow-y-auto rounded-md border border-border bg-card p-3 text-xs shadow-md",
                      children: ($$renderer6) => {
                        $$renderer6.push(`<div class="flex items-center"><span class="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Filter results</span> `);
                        if (activeCount() > 0) {
                          $$renderer6.push("<!--[0-->");
                          $$renderer6.push(`<button type="button" class="ml-auto flex items-center gap-0.5 text-[11px] text-muted-foreground hover:text-foreground">`);
                          X($$renderer6, { class: "size-3" });
                          $$renderer6.push(`<!----> Clear</button>`);
                        } else {
                          $$renderer6.push("<!--[-1-->");
                        }
                        $$renderer6.push(`<!--]--></div> <div class="flex flex-col gap-1.5"><span class="text-muted-foreground">Add a filter</span> <div class="flex gap-1.5">`);
                        $$renderer6.select(
                          {
                            value: colName,
                            "aria-label": "Column",
                            class: "h-8 flex-1 rounded-md border border-border bg-background px-2 text-xs text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                          },
                          ($$renderer7) => {
                            $$renderer7.option({ value: "", disabled: true }, ($$renderer8) => {
                              $$renderer8.push(`Column…`);
                            });
                            $$renderer7.push(`<!--[-->`);
                            const each_array = ensure_array_like(visibleCols());
                            for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
                              let c = each_array[$$index];
                              $$renderer7.option({ value: c.name }, ($$renderer8) => {
                                $$renderer8.push(`${escape_html(c.name)} · ${escape_html(c.type)}`);
                              });
                            }
                            $$renderer7.push(`<!--]-->`);
                          }
                        );
                        $$renderer6.push(` `);
                        $$renderer6.select(
                          {
                            value: op,
                            "aria-label": "Operator",
                            class: "h-8 w-28 rounded-md border border-border bg-background px-2 text-xs text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                          },
                          ($$renderer7) => {
                            $$renderer7.push(`<!--[-->`);
                            const each_array_1 = ensure_array_like(opOptions());
                            for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
                              let o = each_array_1[$$index_1];
                              $$renderer7.option({ value: o.value }, ($$renderer8) => {
                                $$renderer8.push(`${escape_html(o.label)}`);
                              });
                            }
                            $$renderer7.push(`<!--]-->`);
                          }
                        );
                        $$renderer6.push(`</div> <div class="flex gap-1.5">`);
                        Input($$renderer6, {
                          placeholder: colType() === "number" ? "number…" : colType() === "time" ? "date/time…" : colType() === "boolean" ? "true / false" : "value…",
                          class: "h-8 flex-1 text-xs",
                          onkeydown: (e) => {
                            if (e.key === "Enter") {
                              e.preventDefault();
                              addFilter();
                            }
                          },
                          get value() {
                            return val;
                          },
                          set value($$value) {
                            val = $$value;
                            $$settled = false;
                          }
                        });
                        $$renderer6.push(`<!----> `);
                        Button($$renderer6, {
                          type: "button",
                          size: "default",
                          disabled: !colName,
                          onclick: addFilter,
                          children: ($$renderer7) => {
                            Plus($$renderer7, { class: "size-4" });
                            $$renderer7.push(`<!----> Add`);
                          },
                          $$slots: { default: true }
                        });
                        $$renderer6.push(`<!----></div> <span class="text-[10px] text-muted-foreground/70">Pick a column, an operator, a value, then Add — the search re-runs immediately.</span> `);
                        {
                          $$renderer6.push("<!--[-1-->");
                        }
                        $$renderer6.push(`<!--]--></div> <div class="border-t border-border pt-2"><button type="button" class="flex w-full items-center justify-between text-[11px] text-muted-foreground hover:text-foreground"><span>Manage columns (${escape_html(visibleCols().length)}/${escape_html(columns.length)} shown)</span> <span class="text-[10px]">${escape_html("show")}</span></button> `);
                        {
                          $$renderer6.push("<!--[-1-->");
                        }
                        $$renderer6.push(`<!--]--></div> <details class="border-t border-border pt-2"><summary class="cursor-pointer text-[11px] text-muted-foreground hover:text-foreground">Advanced — raw SQL (WHERE)</summary> <textarea${attr("rows", 2)}${attr("placeholder", wherePlaceholder())} class="mt-1.5 min-h-[2rem] w-full resize-y rounded-md border border-border bg-background px-2 py-1.5 font-mono text-[11px] text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">`);
                        const $$body = escape_html(whereSql);
                        if ($$body) {
                          $$renderer6.push(`${$$body}`);
                        }
                        $$renderer6.push(`</textarea> <span class="text-[10px] text-muted-foreground/70">The builder above writes here. Edit directly for OR / parentheses / functions.</span></details>`);
                      },
                      $$slots: { default: true }
                    });
                    $$renderer5.push("<!--]-->");
                  } else {
                    $$renderer5.push("<!--[!-->");
                    $$renderer5.push("<!--]-->");
                  }
                }
              });
              $$renderer4.push("<!--]-->");
            } else {
              $$renderer4.push("<!--[!-->");
              $$renderer4.push("<!--]-->");
            }
          },
          $$slots: { default: true }
        });
        $$renderer3.push("<!--]-->");
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push("<!--]-->");
      }
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { spec });
  });
}
function Help_popover($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { examples } = $$props;
    let open = false;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (Popover) {
        $$renderer3.push("<!--[-->");
        Popover($$renderer3, {
          get open() {
            return open;
          },
          set open($$value) {
            open = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            if (Popover_trigger) {
              $$renderer4.push("<!--[-->");
              Popover_trigger($$renderer4, {
                class: "flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-secondary/40 hover:text-foreground",
                title: "Examples & tips",
                children: ($$renderer5) => {
                  Circle_help($$renderer5, { class: "size-4" });
                },
                $$slots: { default: true }
              });
              $$renderer4.push("<!--]-->");
            } else {
              $$renderer4.push("<!--[!-->");
              $$renderer4.push("<!--]-->");
            }
            $$renderer4.push(` `);
            if (Portal) {
              $$renderer4.push("<!--[-->");
              Portal($$renderer4, {
                children: ($$renderer5) => {
                  if (Popover_content) {
                    $$renderer5.push("<!--[-->");
                    Popover_content($$renderer5, {
                      side: "bottom",
                      align: "end",
                      sideOffset: 6,
                      class: "z-50 w-[min(92vw,400px)] rounded-lg border border-border bg-card p-3 text-xs shadow-md",
                      children: ($$renderer6) => {
                        $$renderer6.push(`<div class="mb-1.5 font-medium text-foreground">Try an example</div> <div class="grid gap-0.5"><!--[-->`);
                        const each_array = ensure_array_like(Object.entries(examples));
                        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
                          let [key, info] = each_array[$$index];
                          $$renderer6.push(`<button type="button" class="grid grid-cols-[84px_1fr] items-baseline gap-2 rounded px-2 py-1.5 text-left transition-colors hover:bg-secondary/50"><span class="font-medium text-foreground">${escape_html(info.label)}</span> <span class="min-w-0"><code class="rounded bg-surface2 px-1.5 py-0.5 font-mono text-[11px] text-primary">${escape_html(info.example)}</code> <span class="mt-0.5 block text-muted-foreground">${escape_html(info.explain)}</span></span></button>`);
                        }
                        $$renderer6.push(`<!--]--></div> <div class="mt-2 flex items-start gap-1.5 rounded border border-dashed border-border bg-muted/40 p-2 text-muted-foreground">`);
                        Image($$renderer6, { class: "mt-0.5 size-3.5 shrink-0" });
                        $$renderer6.push(`<!----> <span><strong class="text-foreground">Attach an image</strong> (📎 or drag in) to add a visual pass.</span></div> <a href="/guide" class="mt-2 flex items-center gap-1.5 rounded px-2 py-1.5 font-medium text-primary transition-colors hover:bg-secondary/50">Full guide: how search works `);
                        Arrow_right($$renderer6, { class: "size-3.5" });
                        $$renderer6.push(`<!----></a>`);
                      },
                      $$slots: { default: true }
                    });
                    $$renderer5.push("<!--]-->");
                  } else {
                    $$renderer5.push("<!--[!-->");
                    $$renderer5.push("<!--]-->");
                  }
                }
              });
              $$renderer4.push("<!--]-->");
            } else {
              $$renderer4.push("<!--[!-->");
              $$renderer4.push("<!--]-->");
            }
          },
          $$slots: { default: true }
        });
        $$renderer3.push("<!--]-->");
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push("<!--]-->");
      }
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}
function Search_settings($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      kind,
      resultN = "100",
      rerank = false,
      rerankN = "20",
      weightPct = null,
      style = "loose",
      sceneMethod = "vector"
    } = $$props;
    const resultOptions = [20, 50, 100, 200].map((n) => ({ value: String(n), label: String(n) }));
    const rerankOptions = [10, 20, 50, 100].map((n) => ({ value: String(n), label: String(n) }));
    const matchOptions = [
      {
        value: "loose",
        label: "Loose",
        description: "Words anywhere in the chunk; stem-aware."
      },
      {
        value: "phrase",
        label: "Phrase",
        description: "Exact words, consecutive order."
      },
      {
        value: "fuzzy",
        label: "Fuzzy",
        description: "Allow up to 2 typos per word."
      }
    ];
    const sceneOptions = [
      {
        value: "vector",
        label: "Meaning",
        description: "Vector search over the caption — semantically similar scenes."
      },
      {
        value: "fts",
        label: "Keyword",
        description: "BM25 over the caption text — exact Swedish words in the scene description."
      }
    ];
    let auto = weightPct === null;
    let weightVal = weightPct ?? 50;
    const balanceLabel = derived(() => auto ? "Auto (RRF)" : weightVal === 50 ? "balanced" : weightVal < 50 ? `${100 - weightVal}% keyword` : `${weightVal}% vector`);
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (Popover) {
        $$renderer3.push("<!--[-->");
        Popover($$renderer3, {
          children: ($$renderer4) => {
            if (Popover_trigger) {
              $$renderer4.push("<!--[-->");
              Popover_trigger($$renderer4, {
                class: "inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-background px-3 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground data-[state=open]:bg-muted data-[state=open]:text-foreground",
                title: "Search settings — result count, reranking, fusion balance, match style",
                children: ($$renderer5) => {
                  Settings_2($$renderer5, { class: "size-3.5" });
                  $$renderer5.push(`<!----> <span>Settings</span>`);
                },
                $$slots: { default: true }
              });
              $$renderer4.push("<!--]-->");
            } else {
              $$renderer4.push("<!--[!-->");
              $$renderer4.push("<!--]-->");
            }
            $$renderer4.push(` `);
            if (Portal) {
              $$renderer4.push("<!--[-->");
              Portal($$renderer4, {
                children: ($$renderer5) => {
                  if (Popover_content) {
                    $$renderer5.push("<!--[-->");
                    Popover_content($$renderer5, {
                      sideOffset: 6,
                      align: "end",
                      class: "z-50 flex w-[320px] flex-col gap-3 rounded-md border border-border bg-card p-4 text-xs shadow-md",
                      children: ($$renderer6) => {
                        Field($$renderer6, {
                          label: "Results to return",
                          inline: true,
                          children: ($$renderer7) => {
                            Select_1($$renderer7, {
                              options: resultOptions,
                              ariaLabel: "Results to return",
                              class: "w-24",
                              get value() {
                                return resultN;
                              },
                              set value($$value) {
                                resultN = $$value;
                                $$settled = false;
                              }
                            });
                          }
                        });
                        $$renderer6.push(`<!----> `);
                        Field($$renderer6, {
                          label: "Rerank results",
                          inline: true,
                          children: ($$renderer7) => {
                            Switch_1($$renderer7, {
                              "aria-label": "Rerank results",
                              get checked() {
                                return rerank;
                              },
                              set checked($$value) {
                                rerank = $$value;
                                $$settled = false;
                              }
                            });
                          }
                        });
                        $$renderer6.push(`<!----> `);
                        if (rerank) {
                          $$renderer6.push("<!--[0-->");
                          Field($$renderer6, {
                            label: "Rerank top",
                            description: "Cross-encoder re-scores this many top results (the rest keep their order). Smaller = faster, more precise head.",
                            inline: true,
                            children: ($$renderer7) => {
                              Select_1($$renderer7, {
                                options: rerankOptions,
                                ariaLabel: "Rerank candidates",
                                class: "w-24",
                                get value() {
                                  return rerankN;
                                },
                                set value($$value) {
                                  rerankN = $$value;
                                  $$settled = false;
                                }
                              });
                            }
                          });
                        } else {
                          $$renderer6.push("<!--[-1-->");
                        }
                        $$renderer6.push(`<!--]--> `);
                        if (kind === "both") {
                          $$renderer6.push("<!--[0-->");
                          $$renderer6.push(`<div class="flex flex-col gap-1.5 border-t border-border pt-3"><div class="flex items-baseline justify-between"><span class="text-xs font-medium text-foreground">Fusion balance</span> <span class="text-[11px] text-muted-foreground">${escape_html(balanceLabel())}</span></div> <label class="flex items-center justify-between"><span class="text-[11px] text-muted-foreground">Auto-fuse (RRF)</span> `);
                          Switch_1($$renderer6, {
                            "aria-label": "Auto-fuse with RRF",
                            get checked() {
                              return auto;
                            },
                            set checked($$value) {
                              auto = $$value;
                              $$settled = false;
                            }
                          });
                          $$renderer6.push(`<!----></label> `);
                          if (!auto) {
                            $$renderer6.push("<!--[0-->");
                            Slider_1($$renderer6, {
                              min: 0,
                              max: 100,
                              step: 5,
                              "aria-label": "Fusion balance",
                              get value() {
                                return weightVal;
                              },
                              set value($$value) {
                                weightVal = $$value;
                                $$settled = false;
                              }
                            });
                            $$renderer6.push(`<!----> <div class="flex justify-between text-[10px] text-muted-foreground"><span>← keyword</span> <span>vector →</span></div>`);
                          } else {
                            $$renderer6.push("<!--[-1-->");
                          }
                          $$renderer6.push(`<!--]--></div>`);
                        } else {
                          $$renderer6.push("<!--[-1-->");
                        }
                        $$renderer6.push(`<!--]--> `);
                        if (kind === "scene") {
                          $$renderer6.push("<!--[0-->");
                          Field($$renderer6, {
                            label: "Scene search",
                            description: "Search the AI caption of each frame by meaning (vector) or exact words (keyword).",
                            class: "border-t border-border pt-3",
                            children: ($$renderer7) => {
                              Radio_group($$renderer7, {
                                options: sceneOptions,
                                get value() {
                                  return sceneMethod;
                                },
                                set value($$value) {
                                  sceneMethod = $$value;
                                  $$settled = false;
                                }
                              });
                            }
                          });
                        } else {
                          $$renderer6.push("<!--[-1-->");
                        }
                        $$renderer6.push(`<!--]--> `);
                        if (kind !== "meaning" && kind !== "scene") {
                          $$renderer6.push("<!--[0-->");
                          Field($$renderer6, {
                            label: "Keyword match style",
                            class: "border-t border-border pt-3",
                            children: ($$renderer7) => {
                              Radio_group($$renderer7, {
                                options: matchOptions,
                                get value() {
                                  return style;
                                },
                                set value($$value) {
                                  style = $$value;
                                  $$settled = false;
                                }
                              });
                            }
                          });
                        } else {
                          $$renderer6.push("<!--[-1-->");
                        }
                        $$renderer6.push(`<!--]--> `);
                        if (voiceSearch.built) {
                          $$renderer6.push("<!--[0-->");
                          $$renderer6.push(`<div class="flex flex-col gap-1 border-t border-border pt-3">`);
                          Field($$renderer6, {
                            label: "Voice: include same video",
                            inline: true,
                            children: ($$renderer7) => {
                              Switch_1($$renderer7, {
                                "aria-label": "Voice results: include the anchor's own video",
                                get checked() {
                                  return voiceSearch.includeSameDoc;
                                },
                                set checked($$value) {
                                  voiceSearch.includeSameDoc = $$value;
                                  $$settled = false;
                                }
                              });
                            }
                          });
                          $$renderer6.push(`<!----> <span class="text-[11px] text-muted-foreground">"Find this voice" normally hides matches from the anchor's own video. Applies
            immediately to an active voice search.</span></div>`);
                        } else {
                          $$renderer6.push("<!--[-1-->");
                        }
                        $$renderer6.push(`<!--]-->`);
                      },
                      $$slots: { default: true }
                    });
                    $$renderer5.push("<!--]-->");
                  } else {
                    $$renderer5.push("<!--[!-->");
                    $$renderer5.push("<!--]-->");
                  }
                }
              });
              $$renderer4.push("<!--]-->");
            } else {
              $$renderer4.push("<!--[!-->");
              $$renderer4.push("<!--]-->");
            }
          },
          $$slots: { default: true }
        });
        $$renderer3.push("<!--]-->");
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push("<!--]-->");
      }
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { resultN, rerank, rerankN, weightPct, style, sceneMethod });
  });
}
function Search_bar($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { spec = void 0, onsubmit } = $$props;
    let kind = spec.mode === "semantic" ? "meaning" : spec.mode === "scene" || spec.mode === "scene_fts" ? "scene" : spec.mode === "fts" ? "keyword" : "both";
    let sceneMethod = spec.mode === "scene_fts" ? "fts" : "vector";
    let style = spec.phrase ? "phrase" : spec.fuzziness === 2 ? "fuzzy" : "loose";
    let rerank = spec.rerank ?? false;
    let rerankN = String(spec.rerankN ?? 20);
    let resultN = String(spec.n ?? 100);
    let weightPct = spec.weight === void 0 || spec.weight === null ? null : Math.round(spec.weight * 100);
    let q = spec.q;
    let qVec = spec.qVec ?? "";
    let imageFile = spec.image ?? null;
    const humanize2 = (key) => key.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    const kindOptions = derived(() => {
      const view = activeView();
      const opts = [];
      if (view.hasMode("fts")) opts.push({ value: "keyword", label: "Keyword" });
      if (view.hasMode("semantic")) opts.push({ value: "meaning", label: "Vector" });
      if (view.hasMode("hybrid")) opts.push({ value: "both", label: "Hybrid" });
      if (view.hasMode("scene")) opts.push({ value: "scene", label: "Scene" });
      const named = /* @__PURE__ */ new Set(["semantic", "visual", "scene"]);
      for (const s of view.vectorSpaces) {
        if (!named.has(s.key) && s.encoder !== "image") {
          opts.push({ value: s.key, label: humanize2(s.key) });
        }
      }
      return opts;
    });
    const imageSpace = derived(() => activeView().vectorSpaces.find((s) => s.encoder === "image") ?? null);
    const hasVisual = derived(() => imageSpace() !== null);
    function buildSpec() {
      let mode;
      const keywordInvolved = kind === "keyword" || kind === "both";
      const hasText = q.trim() !== "" || qVec.trim() !== "";
      if (imageFile && hasText) mode = "all";
      else if (imageFile) mode = imageSpace()?.key ?? "visual";
      else if (kind === "meaning") mode = "semantic";
      else if (kind === "scene") mode = sceneMethod === "fts" ? "scene_fts" : "scene";
      else if (kind === "both") mode = "hybrid";
      else if (kind === "keyword") mode = "fts";
      else mode = kind;
      return {
        q: q.trim(),
        qVec: qVec.trim() || void 0,
        mode,
        phrase: keywordInvolved && style === "phrase",
        fuzziness: keywordInvolved && style === "fuzzy" ? 2 : 0,
        rerank,
        rerankN: rerank ? Number(rerankN) : void 0,
        weight: kind === "both" && weightPct !== null ? weightPct / 100 : void 0,
        n: Number(resultN) || 30,
        image: imageFile,
        // Filter fields are owned by <FilterPopover> on `spec` — pass them
        // through so a new search keeps the active filters (the old buildSpec
        // dropped `where`/`prefilter`, silently discarding the SQL filter).
        filters: spec.filters,
        where: spec.where,
        prefilter: spec.prefilter
      };
    }
    function submit() {
      const next = buildSpec();
      spec = next;
      onsubmit?.(next);
    }
    let fileInput = null;
    let audioInput = null;
    const examples = {
      keyword: {
        label: "Keyword",
        example: "betänkandet",
        explain: 'Match transcripts that CONTAIN your words (in any order). Swedish stemmer also accepts inflections — "betänkandet" finds "betänkande" / "betänkanden" / "betänkandet".'
      },
      phrase: {
        label: "Phrase",
        example: "alkoholmonopolets framtid",
        explain: "Words must appear in this EXACT order, side by side."
      },
      fuzzy: {
        label: "Fuzzy",
        example: "betänkadet",
        explain: "Like Keyword but allows up to 2 letter typos per word — useful when unsure of spelling."
      },
      meaning: {
        label: "Vector",
        example: "klimatkris",
        explain: `Vector search — finds chunks that DISCUSS the topic, even if those exact words aren't there. "klimat" can find "miljö" / "ekosystem".`
      },
      both: {
        label: "Hybrid",
        example: "regeringens beslut",
        explain: "Run Keyword (FTS) AND Vector together, fuse the rankings. Recommended default."
      },
      scene: {
        label: "Scene",
        example: "demonstranter med plakat",
        explain: "Searches the AI-written Swedish caption of each video FRAME — finds clips by what is visible on screen, described in words. Complements Image search (which matches raw visuals)."
      }
    };
    const singlePlaceholder = derived(() => kind === "scene" ? "Describe what's on screen…" : kind === "meaning" ? "Search by meaning…" : "Search transcripts…");
    const summary = derived(() => {
      if (imageFile) {
        if (kind === "meaning" || kind === "both") return "Image + your text → fused over transcript meaning AND visually similar frames.";
        return "Image only → finds visually similar video frames. (Switch to Vector or Hybrid to also use your text.)";
      }
      if (kind === "scene") return examples.scene.explain;
      if (kind === "meaning") return examples.meaning.explain;
      if (kind === "both") return `Hybrid — ${examples.keyword.explain} PLUS ${examples.meaning.explain}`;
      if (style === "phrase") return examples.phrase.explain;
      if (style === "fuzzy") return examples.fuzzy.explain;
      return examples.keyword.explain;
    });
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="px-6 py-3"><form class="flex flex-col gap-2.5"><div class="flex flex-wrap items-center gap-2">`);
      Select_1($$renderer3, {
        options: kindOptions(),
        ariaLabel: "Search mode",
        class: "w-32",
        get value() {
          return kind;
        },
        set value($$value) {
          kind = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----> <span class="hidden flex-1 truncate text-[11px] text-muted-foreground lg:inline">${escape_html(summary())}</span> <div class="ml-auto flex items-center gap-1.5">`);
      if (hasVisual()) {
        $$renderer3.push("<!--[0-->");
        Button($$renderer3, {
          type: "button",
          variant: "outline",
          size: "default",
          title: "Attach an image (drag-drop also works) — search by visual similarity",
          onclick: () => fileInput?.click(),
          children: ($$renderer4) => {
            Paperclip($$renderer4, { class: "size-4" });
            $$renderer4.push(`<!----> Image`);
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!----> <input type="file" accept="image/*" class="hidden"/>`);
      } else {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (voiceSearch.built) {
        $$renderer3.push("<!--[0-->");
        Button($$renderer3, {
          type: "button",
          variant: "outline",
          size: "default",
          disabled: voiceSearch.uploadPending,
          title: "Attach a short audio clip (≤ 25 MB) — find everywhere this voice speaks",
          onclick: () => audioInput?.click(),
          children: ($$renderer4) => {
            if (voiceSearch.uploadPending) {
              $$renderer4.push("<!--[0-->");
              Loader_circle($$renderer4, { class: "size-4 animate-spin" });
            } else {
              $$renderer4.push("<!--[-1-->");
              Audio_lines($$renderer4, { class: "size-4" });
            }
            $$renderer4.push(`<!--]--> Voice`);
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!----> <input type="file" accept="audio/*,video/mp4,.m4a,.mp3,.wav" class="hidden"/>`);
      } else {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--> `);
      Search_settings($$renderer3, {
        kind,
        get resultN() {
          return resultN;
        },
        set resultN($$value) {
          resultN = $$value;
          $$settled = false;
        },
        get rerank() {
          return rerank;
        },
        set rerank($$value) {
          rerank = $$value;
          $$settled = false;
        },
        get rerankN() {
          return rerankN;
        },
        set rerankN($$value) {
          rerankN = $$value;
          $$settled = false;
        },
        get weightPct() {
          return weightPct;
        },
        set weightPct($$value) {
          weightPct = $$value;
          $$settled = false;
        },
        get style() {
          return style;
        },
        set style($$value) {
          style = $$value;
          $$settled = false;
        },
        get sceneMethod() {
          return sceneMethod;
        },
        set sceneMethod($$value) {
          sceneMethod = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----> `);
      Filter_popover($$renderer3, {
        onchange: submit,
        get spec() {
          return spec;
        },
        set spec($$value) {
          spec = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----> `);
      Help_popover($$renderer3, {
        examples
      });
      $$renderer3.push(`<!----></div></div>  `);
      if (kind === "both") {
        $$renderer3.push("<!--[0-->");
        $$renderer3.push(`<div class="flex items-center gap-2">`);
        Input($$renderer3, {
          type: "search",
          class: "h-9 sm:w-52",
          placeholder: "Keyword — exact words",
          "aria-label": "Keyword (FTS)",
          get value() {
            return q;
          },
          set value($$value) {
            q = $$value;
            $$settled = false;
          }
        });
        $$renderer3.push(`<!----> `);
        Input($$renderer3, {
          type: "search",
          class: "h-9 flex-1 sm:max-w-2xl",
          placeholder: "Vector — search by meaning (primary)",
          "aria-label": "Vector — search by meaning",
          get value() {
            return qVec;
          },
          set value($$value) {
            qVec = $$value;
            $$settled = false;
          }
        });
        $$renderer3.push(`<!----> `);
        Button($$renderer3, {
          type: "submit",
          size: "lg",
          children: ($$renderer4) => {
            Search($$renderer4, { class: "size-4" });
            $$renderer4.push(`<!----> Search`);
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!----></div>`);
      } else {
        $$renderer3.push("<!--[-1-->");
        $$renderer3.push(`<div class="flex items-center gap-2">`);
        Input($$renderer3, {
          type: "search",
          class: "h-9 flex-1",
          placeholder: singlePlaceholder(),
          get value() {
            return q;
          },
          set value($$value) {
            q = $$value;
            $$settled = false;
          }
        });
        $$renderer3.push(`<!----> `);
        Button($$renderer3, {
          type: "submit",
          size: "lg",
          children: ($$renderer4) => {
            Search($$renderer4, { class: "size-4" });
            $$renderer4.push(`<!----> Search`);
          },
          $$slots: { default: true }
        });
        $$renderer3.push(`<!----></div>`);
      }
      $$renderer3.push(`<!--]--> `);
      {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--> `);
      {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--> <div class="flex items-baseline gap-2 text-[11px] text-muted-foreground lg:hidden">`);
      if (imageFile) {
        $$renderer3.push("<!--[0-->");
        Image_plus($$renderer3, { class: "size-3.5 self-center text-primary" });
      } else {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--> <span>${escape_html(summary())}</span></div></form></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { spec });
  });
}
function Active_filters($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { spec = void 0, onchange } = $$props;
    const pills = derived(() => {
      const view = activeView();
      const labelFor = (field) => view.metadataFields.find((m) => m.field === field)?.label ?? field;
      const out = [];
      const filters = spec.filters ?? {};
      for (const field of view.filterFields) {
        const value = filters[field];
        if (value) out.push({
          id: `filter:${field}`,
          kind: "filter",
          field,
          label: labelFor(field),
          value
        });
      }
      if (spec.topic) out.push({
        id: "topic",
        kind: "topic",
        label: "Topic",
        value: spec.topic
      });
      if (spec.where) {
        const expr = spec.where.length > 48 ? `${spec.where.slice(0, 48)}…` : spec.where;
        out.push({ id: "where", kind: "where", label: "SQL", value: expr });
      }
      return out;
    });
    if (pills().length) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="flex flex-wrap items-center gap-1.5 px-6 pb-3 text-[11px]"><span class="text-muted-foreground">Active filters:</span> <!--[-->`);
      const each_array = ensure_array_like(pills());
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let p = each_array[$$index];
        $$renderer2.push(`<span class="flex items-center gap-1 rounded-full border border-border bg-secondary px-2 py-0.5 font-medium"><span class="text-muted-foreground">${escape_html(p.label)}:</span> <span class="max-w-[280px] truncate">${escape_html(p.value)}</span> <button type="button"${attr("aria-label", `Remove ${stringify(p.label)} filter`)} class="text-muted-foreground hover:text-destructive">`);
        X($$renderer2, { class: "size-3" });
        $$renderer2.push(`<!----></button></span>`);
      }
      $$renderer2.push(`<!--]--> `);
      if (pills().length > 1) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<button type="button" class="ml-1 text-muted-foreground hover:text-foreground">Clear all</button>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { spec });
  });
}
class FeatureFlags {
  /** True when any /api/chunk-frame fetch has returned 404 — implies
        `extract-chunk-frames` hasn't been run yet. */
  framesUnavailable = false;
}
const features = new FeatureFlags();
const END_SLACK_S = 0.25;
class AudioPreviewStore {
  /** Row key (utils.hitKey) of the clip now playing; null = idle. */
  playing = null;
  /** doc_ids whose media failed to load (404 etc.) — their buttons disable. */
  #failedDocs = new SvelteSet();
  #el = null;
  /** Clip end (s) for the timeupdate guard; Infinity = play to the doc end. */
  #end = Infinity;
  /** doc_id of the loaded source, so the error handler can attribute it. */
  #docId = null;
  isPlaying(key) {
    return this.playing === key;
  }
  /** True when this doc's media previously failed to load — disable the button. */
  isFailed(docId) {
    return this.#failedDocs.has(docId);
  }
  /** Play `[start, end]` of `docId`'s media, keyed by `key` (a row identity).
   *  Calling with the key that is already playing pauses instead — every
   *  button is a toggle. Starting a clip pauses any other row's clip. */
  toggle(clip) {
    const { key, docId, start, end } = clip;
    if (this.playing === key) {
      this.#el?.pause();
      return;
    }
    if (this.#failedDocs.has(docId)) return;
    const el = this.#el ??= this.#create();
    el.pause();
    this.#docId = docId;
    this.#end = end > start ? end : Infinity;
    const view = activeView();
    const src = view.mediaUrl({ [view.docKeyField]: docId });
    el.src = `${src}#t=${start}${end > start ? `,${end}` : ""}`;
    this.playing = key;
    void el.play().catch((err) => {
      if (err instanceof DOMException && err.name === "AbortError") return;
      if (this.playing === key) this.playing = null;
      this.#failedDocs.add(docId);
    });
  }
  #create() {
    const el = new Audio();
    el.preload = "none";
    el.addEventListener("pause", () => {
      if (el.paused) this.playing = null;
    });
    el.addEventListener("ended", () => {
      this.playing = null;
    });
    el.addEventListener("error", () => {
      if (this.#docId) this.#failedDocs.add(this.#docId);
      this.playing = null;
    });
    el.addEventListener("timeupdate", () => {
      if (el.currentTime >= this.#end + END_SLACK_S) el.pause();
    });
    return el;
  }
}
const audioPreview = new AudioPreviewStore();
function Hit_card($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      hit,
      query = "",
      active = false,
      layout = "row",
      highlight: highlightProp,
      onclick
    } = $$props;
    const view = activeView();
    const docId = derived(() => view.docId(hit));
    const title = derived(() => view.title(hit));
    const body = derived(() => view.body(hit));
    const caption = derived(() => view.caption(hit));
    const time = derived(() => view.time(hit));
    const highlight = derived(() => highlightProp ?? makeHighlighter(queryTerms(query)));
    const metaLine = derived(() => {
      const segs = [];
      if (time()) segs.push(`${fmtTime(time().start)} → ${fmtTime(time().end)}`);
      for (const m of view.metadata(hit)) if (m.value !== title()) segs.push(m.value);
      return segs.join("  ·  ");
    });
    const voice = derived(() => isVoiceHit(hit) ? hit : null);
    const band = derived(() => voice() ? voiceBandOf(voice().turn_score) : null);
    const bandTitle = derived(() => voice() ? `Voice similarity ${voice().turn_score.toFixed(3)} (1 − cosine distance). Confidence bands are still calibrating.` : "");
    function voiceMeta($$renderer3) {
      if (voice()) {
        $$renderer3.push("<!--[0-->");
        $$renderer3.push(`<div class="flex flex-wrap items-center gap-1 pt-0.5"><span class="rounded-full border border-border bg-secondary px-1.5 py-px font-mono text-[10px] text-foreground">${escape_html(voice().speaker_label)} · ${escape_html(fmtTime(voice().turn_start))}–${escape_html(fmtTime(voice().turn_end))}</span> `);
        if (band() === "strong") {
          $$renderer3.push("<!--[0-->");
          $$renderer3.push(`<span${attr("title", bandTitle())} class="rounded-full bg-emerald-500/15 px-1.5 py-px text-[10px] font-medium text-emerald-600 dark:text-emerald-400">Strong match</span>`);
        } else if (band() === "possible") {
          $$renderer3.push("<!--[1-->");
          $$renderer3.push(`<span${attr("title", bandTitle())} class="rounded-full bg-amber-500/15 px-1.5 py-px text-[10px] font-medium text-amber-600 dark:text-amber-400">Possible</span>`);
        } else {
          $$renderer3.push("<!--[-1-->");
          $$renderer3.push(`<span${attr("title", bandTitle())} class="rounded-full bg-muted px-1.5 py-px font-mono text-[10px] text-muted-foreground">${escape_html(voice().turn_score.toFixed(2))}</span>`);
        }
        $$renderer3.push(`<!--]--></div>`);
      } else {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]-->`);
    }
    function playButton($$renderer3, extra) {
      const rowKey = hitKey(hit);
      const playing = audioPreview.isPlaying(rowKey);
      const failed = audioPreview.isFailed(docId());
      const clipStart = voice() ? voice().turn_start : time()?.start ?? 0;
      const clipEnd = voice() ? voice().turn_end : time()?.end ?? 0;
      $$renderer3.push(`<button type="button"${attr("disabled", failed, true)}${attr("title", failed ? "Audio unavailable — media failed to load" : playing ? "Pause" : `Play ${fmtTime(clipStart)}–${fmtTime(clipEnd)}`)}${attr("aria-label", playing ? "Pause clip" : "Play clip")}${attr("aria-pressed", playing)}${attr_class(clsx(cn("inline-flex size-6 items-center justify-center rounded-md transition-opacity focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed", extra, playing ? "opacity-100" : "opacity-0 group-hover:opacity-100", failed && "group-hover:opacity-40")))}>`);
      if (playing) {
        $$renderer3.push("<!--[0-->");
        Pause($$renderer3, { class: "size-3.5" });
      } else {
        $$renderer3.push("<!--[-1-->");
        Play($$renderer3, { class: "size-3.5" });
      }
      $$renderer3.push(`<!--]--></button>`);
    }
    if (layout === "tile") {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="group relative h-full"${attr("data-hit-key", hitKey(hit))}><button type="button"${attr("aria-pressed", active)}${attr_class(clsx(cn("flex h-full w-full flex-col overflow-hidden rounded-lg border bg-card text-left transition-all", "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring", active ? "border-primary ring-2 ring-primary shadow-md -translate-y-0.5" : "border-border hover:border-primary hover:shadow-md hover:-translate-y-0.5")))}><div class="relative aspect-video w-full overflow-hidden bg-muted"><img${attr("src", thumbnailUrl(hit))} loading="lazy" alt="" class="h-full w-full object-cover transition-transform group-hover:scale-105" onerror="this.__e=event"/> `);
      if (!features.framesUnavailable) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<img${attr("src", chunkFrameUrl(hit))} loading="lazy" alt="" class="absolute right-1.5 bottom-1.5 h-10 w-16 rounded border border-background bg-black object-cover shadow" onerror="this.__e=event"/>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (time()) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<span class="absolute bottom-1.5 left-1.5 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-white">${escape_html(fmtTime(time().start))}</span>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></div> <div class="flex flex-1 flex-col gap-1 p-2.5"><div class="line-clamp-1 text-xs font-semibold leading-snug">${escape_html(title())}</div> `);
      if (metaLine()) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<div class="truncate font-mono text-[10px] text-muted-foreground">${escape_html(metaLine())}</div>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--> `);
      voiceMeta($$renderer2);
      $$renderer2.push(`<!----> <div class="line-clamp-3 text-xs leading-snug [overflow-wrap:anywhere]">${html(highlight()(body()))}</div> `);
      if (caption()) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<div class="line-clamp-2 text-[10px] italic text-muted-foreground"${attr("title", caption())}>🎬 ${escape_html(caption())}</div>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></div></button> <div class="absolute top-1.5 right-1.5 z-[2] flex gap-1">`);
      playButton($$renderer2, "bg-black/60 text-white enabled:hover:bg-primary");
      $$renderer2.push(`<!----> `);
      if (voiceSearch.built) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<button type="button" title="Find this voice — everywhere this speaker talks, across videos" aria-label="Find this voice" class="inline-flex size-6 items-center justify-center rounded-md bg-black/60 text-white opacity-0 transition-opacity hover:bg-primary focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover:opacity-100">`);
        Audio_lines($$renderer2, { class: "size-3.5" });
        $$renderer2.push(`<!----></button>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></div></div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div class="group relative"${attr("data-hit-key", hitKey(hit))}><button type="button"${attr("aria-pressed", active)}${attr_class(clsx(cn("flex w-full items-start gap-3 border-b border-border px-3 py-2.5 text-left transition-colors", "hover:bg-secondary/40", active && "bg-primary/15 ring-2 ring-inset ring-primary z-[1] relative shadow-[inset_4px_0_0_0_var(--color-primary)]")))}><div class="relative flex-none"><img${attr("src", thumbnailUrl(hit))} loading="lazy" alt="" class="h-[54px] w-[96px] rounded bg-black object-cover" onerror="this.__e=event"/> `);
      if (!features.framesUnavailable) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<img${attr("src", chunkFrameUrl(hit))} loading="lazy" alt="" class="absolute -right-0.5 -bottom-0.5 h-5 w-9 rounded-sm border border-background bg-black object-cover" onerror="this.__e=event"/>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></div> <div class="min-w-0 flex-1 space-y-0.5"><div class="line-clamp-2 text-sm font-semibold leading-snug [overflow-wrap:anywhere]">${escape_html(title())}</div> `);
      if (metaLine()) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<div class="truncate font-mono text-[11px] text-muted-foreground">${escape_html(metaLine())}</div>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--> `);
      voiceMeta($$renderer2);
      $$renderer2.push(`<!----> <div class="line-clamp-3 text-sm leading-snug [overflow-wrap:anywhere]">${html(highlight()(body()))}</div> `);
      if (caption()) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<div class="line-clamp-1 text-[11px] italic text-muted-foreground"${attr("title", caption())}>🎬 ${escape_html(caption())}</div>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></div></button> <div class="absolute top-2 right-2 z-[2] flex gap-1">`);
      playButton($$renderer2, "border border-border bg-card text-muted-foreground shadow-sm enabled:hover:text-primary");
      $$renderer2.push(`<!----> `);
      if (voiceSearch.built) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<button type="button" title="Find this voice — everywhere this speaker talks, across videos" aria-label="Find this voice" class="inline-flex size-6 items-center justify-center rounded-md border border-border bg-card text-muted-foreground opacity-0 shadow-sm transition-opacity hover:text-primary focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover:opacity-100">`);
        Audio_lines($$renderer2, { class: "size-3.5" });
        $$renderer2.push(`<!----></button>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></div></div>`);
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function Hit_list($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      hits,
      query = "",
      active = null,
      onselect,
      emptyMessage = "Enter a query above."
    } = $$props;
    const activeKey = derived(() => active ? hitKey(active) : null);
    const highlight = derived(() => makeHighlighter(queryTerms(query)));
    $$renderer2.push(`<div>`);
    if (hits.length === 0) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="px-4 py-6 text-sm text-muted-foreground">${escape_html(emptyMessage)}</div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<!--[-->`);
      const each_array = ensure_array_like(hits);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let hit = each_array[$$index];
        Hit_card($$renderer2, {
          hit,
          query,
          highlight: highlight(),
          active: activeKey() === hitKey(hit),
          onclick: () => onselect?.(hit)
        });
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
const asStr = (v2) => v2 === null || v2 === void 0 ? null : String(v2);
const asNum = (v2) => typeof v2 === "number" && Number.isFinite(v2) ? v2 : null;
function humanize(field) {
  const parts = field.split(/[_\s]+/).filter((p) => p && p.toLowerCase() !== "id");
  const base = parts.length ? parts.join(" ") : field;
  return base.charAt(0).toUpperCase() + base.slice(1);
}
function isNumericArrow(t) {
  if (!t) return false;
  return t.startsWith("int") || t.startsWith("uint") || t === "float" || t === "double" || t === "halffloat";
}
function arrowTypeOf(view, field) {
  const tables = view.descriptor.tables;
  const rowTable = view.descriptor.declared.search?.row_table;
  const ordered = rowTable && tables[rowTable] ? [tables[rowTable], ...Object.values(tables)] : Object.values(tables);
  for (const t of ordered) {
    const col = t.columns.find((c) => c.name === field);
    if (col) return col.arrow_type;
  }
  return null;
}
function valueColumn(view, field, label) {
  const numeric = isNumericArrow(arrowTypeOf(view, field));
  const col = {
    key: field,
    label,
    numeric,
    render: (h) => asStr(h[field]) ?? ""
  };
  if (numeric) col.sortValue = (h) => asNum(h[field]);
  return col;
}
function TABLE_COLUMNS() {
  const view = activeView();
  const cols = [
    // Per-row audio preview (the shared one-element audioPreview store).
    { key: "play", label: "Play", render: () => "" },
    { key: "thumbnail", label: "Thumb", render: () => "" },
    {
      // Mode-agnostic relevance (higher = better); blank for unranked hits.
      key: "score",
      label: "Relevance",
      numeric: true,
      sortValue: (h) => relevanceOf(h),
      render: (h) => {
        const r = relevanceOf(h);
        return r != null ? r.toFixed(3) : "";
      }
    },
    {
      // Voice-search results only (blank for text hits): the matched diarized
      // turn — per-video speaker label plus the turn's time span.
      key: "speaker",
      label: "Speaker",
      render: (h) => isVoiceHit(h) ? `${h.speaker_label} · ${fmtTime(h.turn_start)}–${fmtTime(h.turn_end)}` : ""
    }
  ];
  for (const field of view.keyFields) cols.push(valueColumn(view, field, humanize(field)));
  if (view.hasTime) {
    cols.push({
      key: "start",
      label: "Start",
      numeric: true,
      sortValue: (h) => view.time(h)?.start ?? null,
      render: (h) => {
        const t = view.time(h);
        return t ? fmtTime(t.start) : "";
      }
    });
    cols.push({
      key: "end",
      label: "End",
      numeric: true,
      sortValue: (h) => view.time(h)?.end ?? null,
      render: (h) => {
        const t = view.time(h);
        return t ? fmtTime(t.end) : "";
      }
    });
    cols.push({
      key: "duration",
      label: "Dur",
      numeric: true,
      sortValue: (h) => view.duration(h),
      render: (h) => {
        const d = view.duration(h);
        return d != null ? fmtTime(d) : "";
      }
    });
  }
  const keyFields = new Set(view.keyFields);
  for (const { field, label } of view.metadataFields) {
    if (keyFields.has(field)) continue;
    cols.push(valueColumn(view, field, label));
  }
  if (view.bodyField) cols.push({ key: "text", label: "Text", render: (h) => view.body(h) });
  if (view.captionField) cols.push({
    key: "caption",
    label: "Caption",
    render: (h) => view.caption(h) ?? ""
  });
  return cols;
}
function cellValue(col, h) {
  if (col.numeric) return col.sortValue ? col.sortValue(h) : null;
  return col.render(h).toLowerCase();
}
function Hit_table($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      hits,
      active,
      onselect,
      visible,
      query = "",
      widths = {},
      wrap = false,
      onwidthchange
      /** Dragged column widths (px) by column key — a dragged width wins over
       *  the default (auto) sizing; absent key = auto. */
      /** Wrap the free-text columns (body / caption / title) instead of truncating. */
      /** A resize-handle drag ended (px) or was double-clicked (null = reset
       *  to auto) — the parent persists alongside the column-visibility prefs. */
    } = $$props;
    const view = activeView();
    const allColumns = derived(TABLE_COLUMNS);
    const cols = derived(() => allColumns().filter((c) => visible.includes(c.key)));
    const terms = derived(() => queryTerms(query));
    const highlight = derived(() => makeHighlighter(terms()));
    const activeKey = derived(() => active ? hitKey(active) : null);
    const HEADER_ROW_HEIGHT_PX = 32;
    const DATALESS_KEYS = ["play", "thumbnail"];
    const SORTABLE = (c) => !DATALESS_KEYS.includes(c.key);
    const wrapKeys = derived(() => {
      const keys = /* @__PURE__ */ new Set();
      if (view.bodyField) keys.add("text");
      if (view.captionField) keys.add("caption");
      for (const f of view.descriptor.declared.display.title) keys.add(f);
      return keys;
    });
    const MIN_COL_PX = 60;
    let drag = null;
    const clampWidth = (w) => Math.max(MIN_COL_PX, Math.round(w));
    const colWidth = (key) => {
      if (drag?.key === key) return drag.width;
      const w = widths[key];
      return w !== void 0 ? clampWidth(w) : void 0;
    };
    let sortKey = null;
    let filters = {};
    function passesFilter(c, h) {
      const raw = filters[c.key]?.trim();
      if (!raw) return true;
      const value = cellValue(c, h);
      if (c.numeric) {
        const min = Number(raw);
        if (Number.isNaN(min)) return true;
        return typeof value === "number" && value >= min;
      }
      return typeof value === "string" && value.includes(raw.toLowerCase());
    }
    const filteredHits = derived(() => hits.filter((h) => cols().every((c) => passesFilter(c, h))));
    const displayedHits = derived(() => {
      return filteredHits();
    });
    $$renderer2.push(`<div class="overflow-x-auto"><table class="w-full border-collapse text-xs"><thead><tr class="sticky top-0 z-10 border-b border-border bg-card text-left text-muted-foreground"><!--[-->`);
    const each_array = ensure_array_like(cols());
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let c = each_array[$$index];
      const w = colWidth(c.key);
      $$renderer2.push(`<th class="relative overflow-hidden px-3 py-2 font-medium whitespace-nowrap"${attr_style(w !== void 0 ? `width:${w}px;min-width:${w}px;max-width:${w}px` : void 0)}>`);
      if (SORTABLE(c)) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<button type="button" class="flex items-center gap-1 hover:text-foreground"${attr("title", `Sort by ${stringify(c.label)}`)}><span>${escape_html(c.label)}</span> `);
        if (sortKey === c.key) {
          $$renderer2.push("<!--[0-->");
          $$renderer2.push(`<span class="text-primary">${escape_html("▼")}</span>`);
        } else {
          $$renderer2.push("<!--[-1-->");
        }
        $$renderer2.push(`<!--]--></button>`);
      } else {
        $$renderer2.push("<!--[-1-->");
        $$renderer2.push(`<span>${escape_html(c.label)}</span>`);
      }
      $$renderer2.push(`<!--]--> <button type="button"${attr("tabindex", -1)}${attr("aria-label", `Resize ${stringify(c.label)} column`)} title="Drag to set column width · double-click to reset"${attr_class("absolute top-0 right-0 z-[1] h-full w-1.5 cursor-col-resize transition-colors hover:bg-primary/50 " + (drag?.key === c.key ? "bg-primary/60" : "bg-transparent"))}></button></th>`);
    }
    $$renderer2.push(`<!--]--></tr><tr class="sticky z-10 border-b border-border bg-card/95"${attr_style(`top: ${stringify(HEADER_ROW_HEIGHT_PX)}px`)}><!--[-->`);
    const each_array_1 = ensure_array_like(cols());
    for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
      let c = each_array_1[$$index_1];
      $$renderer2.push(`<th class="px-2 py-1 font-normal">`);
      if (SORTABLE(c)) {
        $$renderer2.push("<!--[0-->");
        $$renderer2.push(`<input type="text"${attr("inputmode", c.numeric ? "decimal" : "text")}${attr("value", filters[c.key])}${attr("placeholder", c.numeric ? "min ≥" : "filter")}${attr("aria-label", c.numeric ? `Minimum ${c.label}` : `Filter ${c.label}`)} class="w-full rounded border border-border bg-background px-1.5 py-0.5 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none"/>`);
      } else {
        $$renderer2.push("<!--[-1-->");
      }
      $$renderer2.push(`<!--]--></th>`);
    }
    $$renderer2.push(`<!--]--></tr></thead><tbody><!--[-->`);
    const each_array_2 = ensure_array_like(displayedHits());
    for (let $$index_3 = 0, $$length = each_array_2.length; $$index_3 < $$length; $$index_3++) {
      let hit = each_array_2[$$index_3];
      $$renderer2.push(`<tr${attr_class("cursor-pointer border-b border-border/60 hover:bg-secondary/40 " + (activeKey() === hitKey(hit) ? "bg-primary/15 font-medium [box-shadow:inset_3px_0_0_0_var(--color-primary)]" : ""))}><!--[-->`);
      const each_array_3 = ensure_array_like(cols());
      for (let $$index_2 = 0, $$length2 = each_array_3.length; $$index_2 < $$length2; $$index_2++) {
        let c = each_array_3[$$index_2];
        const w = colWidth(c.key);
        if (c.key === "play") {
          $$renderer2.push("<!--[0-->");
          const rowKey = hitKey(hit);
          const playing = audioPreview.isPlaying(rowKey);
          const failed = audioPreview.isFailed(view.docId(hit));
          const clipStart = isVoiceHit(hit) ? hit.turn_start : view.time(hit)?.start ?? 0;
          const clipEnd = isVoiceHit(hit) ? hit.turn_end : view.time(hit)?.end ?? 0;
          $$renderer2.push(`<td class="px-2 py-1 align-top"><button type="button"${attr("disabled", failed, true)}${attr("title", failed ? "Audio unavailable — media failed to load" : playing ? "Pause" : `Play ${fmtTime(clipStart)}–${fmtTime(clipEnd)}`)}${attr("aria-label", playing ? "Pause clip" : "Play clip")}${attr("aria-pressed", playing)}${attr_class("inline-flex size-6 items-center justify-center rounded-md transition-colors enabled:hover:bg-muted enabled:hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40 " + (playing ? "text-primary" : "text-muted-foreground"))}>`);
          if (playing) {
            $$renderer2.push("<!--[0-->");
            Pause($$renderer2, { class: "size-3.5" });
          } else {
            $$renderer2.push("<!--[-1-->");
            Play($$renderer2, { class: "size-3.5" });
          }
          $$renderer2.push(`<!--]--></button></td>`);
        } else if (c.key === "thumbnail") {
          $$renderer2.push("<!--[1-->");
          $$renderer2.push(`<td class="px-3 py-1.5 align-top"><img${attr("src", thumbnailUrl(hit))} loading="lazy" alt="" class="h-9 w-16 rounded bg-muted object-cover" onerror="this.__e=event"/></td>`);
        } else if (c.key === "text") {
          $$renderer2.push("<!--[2-->");
          const body = view.body(hit);
          $$renderer2.push(`<td class="max-w-[32rem] px-3 py-1.5 align-top text-foreground [overflow-wrap:anywhere]"${attr_style(w !== void 0 ? `max-width:${w}px` : void 0)}${attr("title", body)}><div${attr_class(clsx(wrap ? "" : "line-clamp-2"))}>${html(highlight()(body))}</div></td>`);
        } else {
          $$renderer2.push("<!--[-1-->");
          const wraps = wrap && wrapKeys().has(c.key);
          $$renderer2.push(`<td${attr_class("max-w-[28rem] px-3 py-1.5 align-top text-muted-foreground " + (wraps ? "whitespace-normal break-words" : "truncate whitespace-nowrap"))}${attr_style(w !== void 0 ? `max-width:${w}px` : void 0)}${attr("title", c.render(hit))}>${escape_html(c.render(hit))}</td>`);
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--></tr>`);
    }
    $$renderer2.push(`<!--]--></tbody></table></div>`);
  });
}
const SavedColsSchema = v.object({ cols: v.array(v.string()), known: v.array(v.string()) });
const SavedTablePrefsSchema = v.object({
  cols: v.array(v.string()),
  known: v.array(v.string()),
  widths: v.record(v.string(), v.number()),
  wrap: v.boolean()
});
const StringArraySchema = v.array(v.string());
function loadTablePrefs(args) {
  const { storageKey, allKeys, defaults, legacyMergedKey, legacyPlainKey, legacyAppend } = args;
  try {
    const raw = localStorage.getItem(storageKey);
    if (raw) {
      const saved = v.parse(SavedTablePrefsSchema, JSON.parse(raw));
      const fresh = allKeys.filter((k) => !saved.known.includes(k) && defaults.includes(k));
      return { cols: [...saved.cols, ...fresh], widths: saved.widths, wrap: saved.wrap };
    }
    const hasLegacy = legacyMergedKey && localStorage.getItem(legacyMergedKey) !== null || legacyPlainKey && localStorage.getItem(legacyPlainKey) !== null;
    if (hasLegacy && legacyMergedKey) {
      const cols = loadMergedCols({
        storageKey: legacyMergedKey,
        allKeys,
        defaults,
        legacyKey: legacyPlainKey,
        legacyAppend
      });
      const prefs = { cols, widths: {}, wrap: false };
      persistTablePrefs(storageKey, prefs, allKeys);
      return prefs;
    }
  } catch {
  }
  return { cols: [...defaults], widths: {}, wrap: false };
}
function persistTablePrefs(storageKey, prefs, allKeys) {
  try {
    localStorage.setItem(
      storageKey,
      JSON.stringify({ cols: prefs.cols, known: allKeys, widths: prefs.widths, wrap: prefs.wrap })
    );
  } catch {
  }
}
function loadMergedCols(args) {
  const { storageKey, allKeys, defaults, legacyKey, legacyAppend = [] } = args;
  try {
    const raw = localStorage.getItem(storageKey);
    if (raw) {
      const saved = v.parse(SavedColsSchema, JSON.parse(raw));
      const fresh = allKeys.filter((k) => !saved.known.includes(k) && defaults.includes(k));
      return [...saved.cols, ...fresh];
    }
    if (legacyKey) {
      const legacy = localStorage.getItem(legacyKey);
      if (legacy) {
        const cols = v.parse(StringArraySchema, JSON.parse(legacy));
        const migrated = [...cols, ...legacyAppend.filter((k) => !cols.includes(k))];
        persistMergedCols(storageKey, migrated, allKeys);
        return migrated;
      }
    }
  } catch {
  }
  return [...defaults];
}
function persistMergedCols(storageKey, cols, allKeys) {
  try {
    localStorage.setItem(storageKey, JSON.stringify({ cols, known: allKeys }));
  } catch {
  }
}
function loadCols(storageKey, defaults) {
  try {
    const raw = localStorage.getItem(storageKey);
    if (raw) return v.parse(StringArraySchema, JSON.parse(raw));
  } catch {
  }
  return [...defaults];
}
function Doc_tile($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { doc, active = false } = $$props;
    const view = activeView();
    const title = derived(() => view.title(doc));
    const duration = derived(() => view.duration(doc));
    const metaLine = derived(() => view.metadata(doc).filter((m) => m.value !== title()).map((m) => m.value).join("  ·  "));
    $$renderer2.push(`<button type="button"${attr("aria-pressed", active)}${attr_class(clsx(cn("group flex flex-col overflow-hidden rounded-lg border bg-card text-left transition-all", "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring", active ? "border-primary ring-2 ring-primary shadow-md -translate-y-0.5" : "border-border hover:border-primary hover:shadow-md hover:-translate-y-0.5")))}><div class="relative aspect-video w-full overflow-hidden bg-muted"><img${attr("src", thumbnailUrl(doc))} alt="" loading="lazy" class="h-full w-full object-cover transition-transform group-hover:scale-105" onerror="this.__e=event"/> `);
    if (duration() != null) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<span class="absolute bottom-1.5 right-1.5 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-white backdrop-blur-sm">${escape_html(fmtTime(duration()))}</span>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--></div> <div class="flex flex-1 flex-col gap-1 p-3"><div class="line-clamp-2 text-sm font-medium leading-snug">${escape_html(title())}</div> `);
    if (metaLine()) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="truncate font-mono text-[10px] text-muted-foreground"${attr("title", metaLine())}>${escape_html(metaLine())}</div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--></div></button>`);
  });
}
const EMPTY = /* @__PURE__ */ new Set();
class CrossFilter {
  // All collection fields are `$state.raw` — every mutator below REPLACES the
  // collection (never mutates in place), so raw both works and documents the
  // replace-don't-mutate contract.
  /** Point indices the user picked on the map (untruncated — drives dimming). */
  selectedIds = /* @__PURE__ */ new Set();
  /** Point indices of the current search+facet results; null ⇒ no active filter. */
  filteredIds = null;
  /** Which projection the map is showing; the page may read it to fetch per space. */
  space = "text";
  /** Colour channel for the scatter. */
  colorBy = "cluster";
  /** Per-colour-mode hidden codes (a code only means something within its own
   *  colour channel). Keyed by ColorBy; recoloured to background on the map. */
  hiddenByMode = /* @__PURE__ */ new Map();
  /** Descriptor-identity → index for the CURRENT space (rebuilt on space swap). */
  keyToIndex = /* @__PURE__ */ new Map();
  get hasSelection() {
    return this.selectedIds.size > 0;
  }
  get hasFilter() {
    return this.filteredIds !== null;
  }
  /** Hidden codes for the CURRENT colour mode (shared EMPTY set if none). */
  get hidden() {
    return this.hiddenByMode.get(this.colorBy) ?? EMPTY;
  }
  // ── setters ──────────────────────────────────────────────────────────────
  /** Project search Hit[] to point indices via the current key→index map. */
  setFilteredFromHits(hits) {
    const map = this.keyToIndex;
    const next = /* @__PURE__ */ new Set();
    for (const h of hits) {
      const i = map.get(hitKey(h));
      if (i !== void 0) next.add(i);
    }
    this.filteredIds = next;
  }
  /** Clear the search→map filter (back to the whole corpus). */
  clearFilter() {
    this.filteredIds = null;
  }
  /** Replace the map selection with an explicit FULL index set (no truncation). */
  setSelectedIndices(idx) {
    this.selectedIds = new Set(idx);
  }
  clearSelection() {
    if (this.selectedIds.size > 0) this.selectedIds = /* @__PURE__ */ new Set();
  }
  /** Select every point in a cluster — a first-class lasso-equivalent.
   *  `clusters` is `ArrayLike<number>` (an `Int32Array` from the points payload). */
  selectCluster(clusterId, clusters) {
    const next = /* @__PURE__ */ new Set();
    for (let i = 0; i < clusters.length; i++) {
      if (clusters[i] === clusterId) next.add(i);
    }
    this.selectedIds = next;
  }
  setSpace(s) {
    this.space = s;
  }
  /** Toggle a code's visibility on the map for the CURRENT colour mode
   *  (reassigns Set + Map so reactivity fires). */
  toggleHidden(code) {
    const mode = this.colorBy;
    const cur = this.hiddenByMode.get(mode) ?? EMPTY;
    const next = new Set(cur);
    if (next.has(code)) next.delete(code);
    else next.add(code);
    const nextMap = new Map(this.hiddenByMode);
    if (next.size > 0) nextMap.set(mode, next);
    else nextMap.delete(mode);
    this.hiddenByMode = nextMap;
  }
  /** Un-hide every code in the CURRENT colour mode. */
  showAll() {
    const mode = this.colorBy;
    if (!this.hiddenByMode.has(mode)) return;
    const nextMap = new Map(this.hiddenByMode);
    nextMap.delete(mode);
    this.hiddenByMode = nextMap;
  }
  /** Reset everything tied to the loaded universe (called on a space swap). */
  resetForSpace(keyToIndex) {
    this.keyToIndex = keyToIndex;
    this.selectedIds = /* @__PURE__ */ new Set();
    this.filteredIds = null;
    this.hiddenByMode = /* @__PURE__ */ new Map();
  }
}
const crossFilter = new CrossFilter();
function AtlasMap($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      active = null,
      onSeedSearch,
      onSelectionHits
      /** Promote the current map selection to a search (by stable `_rowid`). */
      /** Surface the lasso/box selection's hits to the page (drives HitTable). */
    } = $$props;
    const view = activeView();
    const hasAtlas = derived(() => view.atlasSpaces.length > 0);
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (!hasAtlas()) {
        $$renderer3.push("<!--[0-->");
        $$renderer3.push(`<div class="grid h-full place-items-center p-6 text-center text-sm text-muted-foreground">This dataset has no embedding map.</div>`);
      } else {
        $$renderer3.push("<!--[-1-->");
        $$renderer3.push(`<div role="presentation" class="relative h-full min-h-0 bg-background">`);
        {
          $$renderer3.push("<!--[0-->");
          $$renderer3.push(`<div class="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">`);
          Loader_circle($$renderer3, { class: "size-4 animate-spin" });
          $$renderer3.push(`<!----> Loading embedding map…</div>`);
        }
        $$renderer3.push(`<!--]--></div>`);
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { active });
  });
}
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const ds = activeView();
    ds.descriptor.declared.time;
    const PAGE_STEP = 30;
    let spec = { q: "", n: 100, mode: "fts" };
    let hits = [];
    let active = null;
    const activeKey = derived(() => active ? hitKey(active) : null);
    let loadingHits = false;
    let loadingMore = false;
    let error = null;
    let allLoaded = false;
    let voiceActive = false;
    let voiceQuery = null;
    let voiceLabel = null;
    let voiceN = PAGE_STEP;
    const VOICE_MAX_N = 100;
    let voiceRawCount = 0;
    let voiceAnchor = null;
    let voiceUpload = null;
    let prevResults = null;
    const resultsQuery = derived(() => voiceActive ? "" : spec.q);
    const gridHighlight = derived(() => makeHighlighter(queryTerms(resultsQuery())));
    const docCount = derived(() => new Set(hits.map((h) => ds.docId(h))).size);
    const PER_PAGE = 24;
    let docs = [];
    let docsTotal = 0;
    let docsPage = 1;
    let view = "list";
    let mapHits = [];
    let mapSelectionTotal = 0;
    const META_KEYS = ds.metadataFields.map((m) => m.field);
    const MAP_TABLE_COLS = [
      "play",
      "thumbnail",
      "score",
      ...META_KEYS.slice(0, 1),
      ...ds.hasTime ? ["start"] : [],
      ...ds.bodyField ? [ds.bodyField] : []
    ];
    const mapTableHits = derived(() => mapSelectionTotal > 0 ? mapHits : hits);
    function onMapSelectionHits(h, total) {
      mapHits = h;
      mapSelectionTotal = total;
    }
    async function seedSearchFromSelection(rowids) {
      loadingHits = true;
      error = null;
      try {
        hits = await getAtlasChunks(rowids);
        active = null;
        allLoaded = true;
        view = "list";
      } catch (e) {
        error = e instanceof Error ? e.message : "failed to load selection";
      } finally {
        loadingHits = false;
      }
    }
    const TABLE_COL_KEYS = TABLE_COLUMNS().map((c) => c.key);
    const DEFAULT_TABLE_COLS = [
      "play",
      "thumbnail",
      "score",
      ...META_KEYS,
      "speaker",
      ...ds.hasTime ? ["start", "end", "duration"] : [],
      ...ds.bodyField ? [ds.bodyField] : [],
      ...ds.captionField ? [ds.captionField] : []
    ];
    const TABLE_PREFS_KEY = "lance-media-table-cols-v6";
    const initialTablePrefs = loadTablePrefs({
      storageKey: TABLE_PREFS_KEY,
      allKeys: TABLE_COL_KEYS,
      defaults: DEFAULT_TABLE_COLS,
      legacyMergedKey: "lance-media-table-cols-v5",
      legacyPlainKey: "lance-media-table-cols-v4",
      legacyAppend: ["speaker"]
    });
    let tableCols = initialTablePrefs.cols;
    let tableWidths = initialTablePrefs.widths;
    let tableWrap = initialTablePrefs.wrap;
    function persistTable() {
      persistTablePrefs(TABLE_PREFS_KEY, { cols: tableCols, widths: tableWidths, wrap: tableWrap }, TABLE_COL_KEYS);
    }
    function setColWidth(key, width) {
      tableWidths = width === null ? Object.fromEntries(Object.entries(tableWidths).filter(([k]) => k !== key)) : { ...tableWidths, [key]: width };
      persistTable();
    }
    function setTableWrap(v2) {
      tableWrap = v2;
      persistTable();
    }
    const DOC_COLUMNS = [
      { key: "thumbnail", label: "Thumb", get: () => "" },
      ...ds.metadataFields.map(({ field, label }) => ({
        key: field,
        label,
        get: (d) => {
          const v2 = d[field];
          return v2 == null ? "" : String(v2);
        }
      })),
      ...ds.hasTime ? [
        {
          key: "duration",
          label: "Dur",
          get: (d) => {
            const dur = ds.duration(d);
            return dur != null ? fmtTime(dur) : "";
          }
        }
      ] : []
    ];
    const DOC_COLS_KEY = "lance-media-doc-cols-v1";
    let docTableCols = loadCols(DOC_COLS_KEY, DOC_COLUMNS.map((c) => c.key));
    function loadGridCols() {
      const v2 = localStorage.getItem("lance-media-gridcols");
      return v2 ? Math.max(2, Math.min(6, Number(v2) || 3)) : 3;
    }
    let gridCols = loadGridCols();
    function setGridCols(n) {
      gridCols = n;
      try {
        localStorage.setItem("lance-media-gridcols", String(n));
      } catch {
      }
    }
    const activeDocId = derived(() => active ? ds.docId(active) : null);
    const isBrowsing = derived(() => !spec.q && !spec.image && !spec.topic && !voiceActive);
    const docsTotalPages = derived(() => Math.max(1, Math.ceil(docsTotal / PER_PAGE)));
    let searchSeq = 0;
    function dedupeVoiceHits(vh) {
      const seen = /* @__PURE__ */ new Set();
      return vh.filter((h) => {
        const k = hitKey(h);
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      });
    }
    function leaveVoiceMode() {
      voiceActive = false;
      voiceQuery = null;
      voiceLabel = null;
      voiceAnchor = null;
      voiceUpload = null;
      voiceN = PAGE_STEP;
      voiceRawCount = 0;
      if (prevResults) {
        hits = prevResults.hits;
        active = prevResults.active;
        allLoaded = prevResults.allLoaded;
        prevResults = null;
      }
    }
    voiceSearch.includeSameDoc;
    async function runSearch(next) {
      const seq = ++searchSeq;
      prevResults = null;
      leaveVoiceMode();
      spec = { ...next, n: next.n ?? PAGE_STEP };
      allLoaded = false;
      if (!spec.q && !spec.image && !spec.topic) {
        hits = [];
        loadingHits = false;
        return;
      }
      crossFilter.clearSelection();
      mapHits = [];
      mapSelectionTotal = 0;
      loadingHits = true;
      error = null;
      try {
        const requested = spec.n ?? PAGE_STEP;
        const result = await search(spec);
        if (seq !== searchSeq) return;
        hits = result;
        active = null;
        if (result.length < requested) allLoaded = true;
      } catch (e) {
        if (seq !== searchSeq) return;
        hits = [];
        error = e instanceof ApiError ? e.detail : e instanceof Error ? e.message : "unknown error";
      } finally {
        if (seq === searchSeq) loadingHits = false;
      }
    }
    async function loadMore() {
      if (loadingMore || allLoaded) return;
      loadingMore = true;
      const seq = ++searchSeq;
      try {
        if (voiceActive && (voiceAnchor || voiceUpload)) {
          const nextN2 = voiceN + PAGE_STEP;
          const res = voiceUpload ? await voiceSimilarUpload(voiceUpload, { n: nextN2 }) : voiceAnchor ? await voiceSimilar(voiceAnchor, { n: nextN2, excludeSameDoc: !voiceSearch.includeSameDoc }) : null;
          if (res && seq === searchSeq) {
            hits = dedupeVoiceHits(res.hits);
            voiceN = nextN2;
            if (res.hits.length <= voiceRawCount || nextN2 >= VOICE_MAX_N) allLoaded = true;
            voiceRawCount = res.hits.length;
          }
          return;
        }
        const nextN = (spec.n ?? PAGE_STEP) + PAGE_STEP;
        const result = await search({ ...spec, n: nextN });
        if (seq === searchSeq) {
          hits = result;
          spec = { ...spec, n: nextN };
          if (result.length < nextN) allLoaded = true;
        }
      } catch {
      } finally {
        loadingMore = false;
      }
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="grid h-full grid-rows-[auto_1fr] min-h-0"><div class="border-b border-border bg-card/40">`);
      Search_bar($$renderer3, {
        onsubmit: runSearch,
        get spec() {
          return spec;
        },
        set spec($$value) {
          spec = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----> `);
      Active_filters($$renderer3, {
        onchange: runSearch,
        get spec() {
          return spec;
        },
        set spec($$value) {
          spec = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----> `);
      if (voiceActive) {
        $$renderer3.push("<!--[0-->");
        $$renderer3.push(`<div class="flex flex-wrap items-center gap-2 px-6 pb-3 text-[11px]"><span class="flex items-center gap-1.5 rounded-md border border-primary bg-primary/10 py-1 pr-1 pl-2 font-medium text-foreground">`);
        Audio_lines($$renderer3, { class: "size-3.5 text-primary" });
        $$renderer3.push(`<!----> <span class="max-w-[24rem] truncate">Voice: ${escape_html(voiceLabel ?? voiceQuery?.doc_id ?? "…")} `);
        if (voiceQuery?.speaker_label) {
          $$renderer3.push("<!--[0-->");
          $$renderer3.push(`· ${escape_html(voiceQuery.speaker_label)}`);
        } else {
          $$renderer3.push("<!--[-1-->");
        }
        $$renderer3.push(`<!--]--></span> <button type="button" title="Clear voice search — back to the previous results" aria-label="Clear voice search" class="inline-flex size-5 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">`);
        X($$renderer3, { class: "size-3.5" });
        $$renderer3.push(`<!----></button></span> `);
        if (voiceQuery && voiceQuery.turn_start != null && voiceQuery.turn_end != null) {
          $$renderer3.push("<!--[0-->");
          $$renderer3.push(`<span class="text-muted-foreground">anchor turn ${escape_html(fmtTime(voiceQuery.turn_start))}–${escape_html(fmtTime(voiceQuery.turn_end))}</span>`);
        } else {
          $$renderer3.push("<!--[-1-->");
        }
        $$renderer3.push(`<!--]--></div>`);
      } else {
        $$renderer3.push("<!--[-1-->");
      }
      $$renderer3.push(`<!--]--></div> `);
      {
        let left = function($$renderer4) {
          $$renderer4.push(`<div class="flex h-full min-h-0 flex-col"><div class="flex items-center gap-3 border-b border-border bg-card/30 px-4 py-2 text-xs"><span class="text-muted-foreground">`);
          if (isBrowsing()) {
            $$renderer4.push("<!--[0-->");
            $$renderer4.push(`Browsing ${escape_html(docsTotal)} document${escape_html("s")}`);
          } else if (loadingHits) {
            $$renderer4.push("<!--[1-->");
            $$renderer4.push(`Searching…`);
          } else if (error) {
            $$renderer4.push("<!--[2-->");
            $$renderer4.push(`<span class="text-destructive">Error: ${escape_html(error)}</span>`);
          } else if (hits.length === 0) {
            $$renderer4.push("<!--[3-->");
            $$renderer4.push(`No hits.`);
          } else {
            $$renderer4.push("<!--[-1-->");
            $$renderer4.push(`<strong class="text-foreground">${escape_html(hits.length)}</strong> ${escape_html(hits.length === 1 ? "chunk" : "chunks")}
              across <strong class="text-foreground">${escape_html(docCount())}</strong> ${escape_html(docCount() === 1 ? "document" : "documents")} `);
            if (allLoaded) {
              $$renderer4.push("<!--[0-->");
              $$renderer4.push(`<span class="text-muted-foreground/70">· all results shown</span>`);
            } else {
              $$renderer4.push("<!--[-1-->");
            }
            $$renderer4.push(`<!--]-->`);
          }
          $$renderer4.push(`<!--]--></span> <div class="ml-auto flex items-center gap-2">`);
          if (view === "grid") {
            $$renderer4.push("<!--[0-->");
            $$renderer4.push(`<div class="flex items-center gap-1 border-r border-border pr-2 mr-1"><span class="text-muted-foreground/70 mr-1">cols</span> `);
            Button($$renderer4, {
              variant: "ghost",
              size: "icon",
              disabled: gridCols <= 2,
              title: "Fewer columns",
              onclick: () => setGridCols(Math.max(2, gridCols - 1)),
              children: ($$renderer5) => {
                Minus($$renderer5, { class: "size-3.5" });
              },
              $$slots: { default: true }
            });
            $$renderer4.push(`<!----> <span class="w-4 text-center font-mono text-[11px]">${escape_html(gridCols)}</span> `);
            Button($$renderer4, {
              variant: "ghost",
              size: "icon",
              disabled: gridCols >= 6,
              title: "More columns",
              onclick: () => setGridCols(Math.min(6, gridCols + 1)),
              children: ($$renderer5) => {
                Plus($$renderer5, { class: "size-3.5" });
              },
              $$slots: { default: true }
            });
            $$renderer4.push(`<!----></div>`);
          } else {
            $$renderer4.push("<!--[-1-->");
          }
          $$renderer4.push(`<!--]--> `);
          Button($$renderer4, {
            variant: view === "list" ? "secondary" : "ghost",
            size: "icon",
            title: "List view",
            onclick: () => view = "list",
            children: ($$renderer5) => {
              List($$renderer5, { class: "size-4" });
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----> `);
          Button($$renderer4, {
            variant: view === "grid" ? "secondary" : "ghost",
            size: "icon",
            title: "Grid view",
            onclick: () => view = "grid",
            children: ($$renderer5) => {
              Layout_grid($$renderer5, { class: "size-4" });
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----> `);
          Button($$renderer4, {
            variant: view === "table" ? "secondary" : "ghost",
            size: "icon",
            title: "Table view — see column values per row",
            onclick: () => view = "table",
            children: ($$renderer5) => {
              Table($$renderer5, { class: "size-4" });
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----> `);
          Button($$renderer4, {
            variant: view === "map" ? "secondary" : "ghost",
            size: "icon",
            title: "Map view — the EVōC embedding atlas (cross-filters with search)",
            onclick: () => view = "map",
            children: ($$renderer5) => {
              Map$1($$renderer5, { class: "size-4" });
            },
            $$slots: { default: true }
          });
          $$renderer4.push(`<!----></div></div> `);
          if (view === "map") {
            $$renderer4.push("<!--[0-->");
            $$renderer4.push(`<div class="min-h-0 flex-1">`);
            {
              let left2 = function($$renderer5) {
                AtlasMap($$renderer5, {
                  onSeedSearch: seedSearchFromSelection,
                  onSelectionHits: onMapSelectionHits,
                  get active() {
                    return active;
                  },
                  set active($$value) {
                    active = $$value;
                    $$settled = false;
                  }
                });
              }, right2 = function($$renderer5) {
                $$renderer5.push(`<div class="flex h-full min-h-0 flex-col border-t border-border bg-card/30"><div class="flex items-center gap-2 border-b border-border px-3 py-1.5 text-xs">`);
                if (mapSelectionTotal > 0) {
                  $$renderer5.push("<!--[0-->");
                  $$renderer5.push(`<span class="font-medium text-foreground">Map selection</span> <span class="text-muted-foreground">${escape_html(mapSelectionTotal.toLocaleString())} chunks `);
                  if (mapSelectionTotal > mapHits.length) {
                    $$renderer5.push("<!--[0-->");
                    $$renderer5.push(`<span class="text-muted-foreground/70">· showing ${escape_html(mapHits.length)}</span>`);
                  } else {
                    $$renderer5.push("<!--[-1-->");
                  }
                  $$renderer5.push(`<!--]--></span>`);
                } else if (hits.length > 0) {
                  $$renderer5.push("<!--[1-->");
                  $$renderer5.push(`<span class="font-medium text-foreground">Search results</span> <span class="text-muted-foreground">${escape_html(hits.length.toLocaleString())} hits · highlighted on the map</span>`);
                } else {
                  $$renderer5.push("<!--[-1-->");
                  $$renderer5.push(`<span class="font-medium text-foreground">Selection</span> <span class="text-muted-foreground">lasso a region, click a legend, or search to list chunks</span>`);
                }
                $$renderer5.push(`<!--]--></div> <div class="min-h-0 flex-1 overflow-auto">`);
                if (mapTableHits().length) {
                  $$renderer5.push("<!--[0-->");
                  Hit_table($$renderer5, {
                    hits: mapTableHits(),
                    active,
                    visible: MAP_TABLE_COLS,
                    widths: tableWidths,
                    wrap: tableWrap,
                    onwidthchange: setColWidth,
                    query: mapSelectionTotal > 0 ? "" : resultsQuery(),
                    onselect: (h) => active = h
                  });
                } else {
                  $$renderer5.push("<!--[-1-->");
                }
                $$renderer5.push(`<!--]--></div></div>`);
              };
              Resizable_split($$renderer4, {
                orientation: "vertical",
                storageKey: "lance-media-search-map-vsplit",
                minLeft: 220,
                minRight: 120,
                initial: 0.66,
                left: left2,
                right: right2
              });
            }
            $$renderer4.push(`<!----></div>`);
          } else {
            $$renderer4.push("<!--[-1-->");
            $$renderer4.push(`<div class="relative min-h-0 flex-1 overflow-y-auto">`);
            if (isBrowsing()) {
              $$renderer4.push("<!--[0-->");
              if (view === "grid") {
                $$renderer4.push("<!--[2-->");
                $$renderer4.push(`<div class="grid gap-4 p-4"${attr_style("", {
                  "grid-template-columns": `repeat(${stringify(gridCols)}, minmax(0, 1fr))`
                })}><!--[-->`);
                const each_array = ensure_array_like(docs);
                for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
                  let doc = each_array[$$index];
                  Doc_tile($$renderer4, {
                    doc,
                    active: activeDocId() === ds.docId(doc)
                  });
                }
                $$renderer4.push(`<!--]--></div>`);
              } else if (view === "table") {
                $$renderer4.push("<!--[3-->");
                const docCols = DOC_COLUMNS.filter((c) => docTableCols.includes(c.key));
                $$renderer4.push(`<div class="flex flex-wrap items-center gap-1 border-b border-border bg-card/30 px-3 py-2 text-[11px]"><span class="mr-1 text-muted-foreground">Columns:</span> <!--[-->`);
                const each_array_1 = ensure_array_like(DOC_COLUMNS);
                for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
                  let c = each_array_1[$$index_1];
                  $$renderer4.push(`<button type="button"${attr_class("rounded border px-1.5 py-0.5 transition-colors " + (docTableCols.includes(c.key) ? "border-primary bg-primary/10 text-foreground" : "border-border text-muted-foreground hover:text-foreground"))}>${escape_html(c.label)}</button>`);
                }
                $$renderer4.push(`<!--]--></div> <div class="overflow-x-auto"><table class="w-full border-collapse text-xs"><thead><tr class="sticky top-0 z-10 border-b border-border bg-card text-left text-muted-foreground"><!--[-->`);
                const each_array_2 = ensure_array_like(docCols);
                for (let $$index_2 = 0, $$length = each_array_2.length; $$index_2 < $$length; $$index_2++) {
                  let c = each_array_2[$$index_2];
                  $$renderer4.push(`<th class="px-3 py-2 font-medium whitespace-nowrap">${escape_html(c.label)}</th>`);
                }
                $$renderer4.push(`<!--]--></tr></thead><tbody><!--[-->`);
                const each_array_3 = ensure_array_like(docs);
                for (let $$index_4 = 0, $$length = each_array_3.length; $$index_4 < $$length; $$index_4++) {
                  let doc = each_array_3[$$index_4];
                  $$renderer4.push(`<tr${attr_class("cursor-pointer border-b border-border/60 hover:bg-secondary/40 " + (activeDocId() === ds.docId(doc) ? "bg-primary/15 font-medium [box-shadow:inset_3px_0_0_0_var(--color-primary)]" : ""))}><!--[-->`);
                  const each_array_4 = ensure_array_like(docCols);
                  for (let $$index_3 = 0, $$length2 = each_array_4.length; $$index_3 < $$length2; $$index_3++) {
                    let c = each_array_4[$$index_3];
                    if (c.key === "thumbnail") {
                      $$renderer4.push("<!--[0-->");
                      $$renderer4.push(`<td class="px-3 py-1.5 align-top"><img${attr("src", thumbnailUrl(doc))} loading="lazy" alt="" class="h-9 w-16 rounded bg-muted object-cover" onerror="this.__e=event"/></td>`);
                    } else {
                      $$renderer4.push("<!--[-1-->");
                      $$renderer4.push(`<td class="max-w-[28rem] truncate px-3 py-1.5 align-top whitespace-nowrap text-muted-foreground"${attr("title", c.get(doc))}>${escape_html(c.get(doc))}</td>`);
                    }
                    $$renderer4.push(`<!--]-->`);
                  }
                  $$renderer4.push(`<!--]--></tr>`);
                }
                $$renderer4.push(`<!--]--></tbody></table></div>`);
              } else {
                $$renderer4.push("<!--[-1-->");
                $$renderer4.push(`<ul class="divide-y divide-border"><!--[-->`);
                const each_array_5 = ensure_array_like(docs);
                for (let $$index_5 = 0, $$length = each_array_5.length; $$index_5 < $$length; $$index_5++) {
                  let doc = each_array_5[$$index_5];
                  $$renderer4.push(`<li><button type="button" class="flex w-full items-center gap-3 px-4 py-2 text-left hover:bg-secondary/40"><span class="flex-1 truncate text-sm">${escape_html(ds.title(doc))}</span> <span class="font-mono text-[11px] text-muted-foreground">${escape_html(ds.metadata(doc).find((m) => m.value !== ds.title(doc))?.value ?? "")}</span></button></li>`);
                }
                $$renderer4.push(`<!--]--></ul>`);
              }
              $$renderer4.push(`<!--]--> `);
              if (docsTotalPages() > 1) {
                $$renderer4.push("<!--[0-->");
                $$renderer4.push(`<div class="sticky bottom-0 flex items-center justify-end gap-1 border-t border-border bg-card/80 px-4 py-2 text-xs backdrop-blur"><span class="mr-2 text-muted-foreground">page ${escape_html(docsPage)} / ${escape_html(docsTotalPages())}</span> `);
                Button($$renderer4, {
                  variant: "outline",
                  size: "icon",
                  disabled: docsPage <= 1,
                  onclick: () => docsPage = Math.max(1, docsPage - 1),
                  children: ($$renderer5) => {
                    Chevron_left($$renderer5, { class: "size-4" });
                  },
                  $$slots: { default: true }
                });
                $$renderer4.push(`<!----> `);
                Button($$renderer4, {
                  variant: "outline",
                  size: "icon",
                  disabled: docsPage >= docsTotalPages(),
                  onclick: () => docsPage = Math.min(docsTotalPages(), docsPage + 1),
                  children: ($$renderer5) => {
                    Chevron_right($$renderer5, { class: "size-4" });
                  },
                  $$slots: { default: true }
                });
                $$renderer4.push(`<!----></div>`);
              } else {
                $$renderer4.push("<!--[-1-->");
              }
              $$renderer4.push(`<!--]-->`);
            } else if (loadingHits) {
              $$renderer4.push("<!--[1-->");
              $$renderer4.push(`<div class="flex h-full items-center justify-center text-sm text-muted-foreground">`);
              Loader_circle($$renderer4, { class: "size-4 animate-spin mr-2" });
              $$renderer4.push(`<!----> Searching…</div>`);
            } else if (hits.length === 0) {
              $$renderer4.push("<!--[2-->");
              $$renderer4.push(`<div class="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-sm text-muted-foreground">`);
              Search_x($$renderer4, { class: "size-6 text-muted-foreground/60" });
              $$renderer4.push(`<!----> <div>No hits.</div> <div class="text-xs">`);
              if (voiceActive) {
                $$renderer4.push("<!--[0-->");
                $$renderer4.push(`No similar voices found — try <strong>include same video</strong> in Settings.`);
              } else {
                $$renderer4.push("<!--[-1-->");
                $$renderer4.push(`Try toggling <strong>Match by</strong> to <em>Semantic</em> or switching <strong>Style</strong> to <em>Fuzzy</em>.`);
              }
              $$renderer4.push(`<!--]--></div></div>`);
            } else if (view === "grid") {
              $$renderer4.push("<!--[3-->");
              $$renderer4.push(`<div class="grid gap-3 p-3"${attr_style("", {
                "grid-template-columns": `repeat(${stringify(gridCols)}, minmax(0, 1fr))`
              })}><!--[-->`);
              const each_array_6 = ensure_array_like(hits);
              for (let $$index_6 = 0, $$length = each_array_6.length; $$index_6 < $$length; $$index_6++) {
                let hit = each_array_6[$$index_6];
                Hit_card($$renderer4, {
                  hit,
                  query: resultsQuery(),
                  highlight: gridHighlight(),
                  active: activeKey() === hitKey(hit),
                  layout: "tile",
                  onclick: () => active = hit
                });
              }
              $$renderer4.push(`<!--]--></div>`);
            } else if (view === "table") {
              $$renderer4.push("<!--[4-->");
              var bind_get = () => tableWrap;
              var bind_set = setTableWrap;
              $$renderer4.push(`<div class="flex flex-wrap items-center gap-1 border-b border-border bg-card/30 px-3 py-2 text-[11px]"><span class="mr-1 text-muted-foreground">Columns:</span> <!--[-->`);
              const each_array_7 = ensure_array_like(TABLE_COLUMNS());
              for (let $$index_7 = 0, $$length = each_array_7.length; $$index_7 < $$length; $$index_7++) {
                let c = each_array_7[$$index_7];
                $$renderer4.push(`<button type="button"${attr_class("rounded border px-1.5 py-0.5 transition-colors " + (tableCols.includes(c.key) ? "border-primary bg-primary/10 text-foreground" : "border-border text-muted-foreground hover:text-foreground"))}>${escape_html(c.label)}</button>`);
              }
              $$renderer4.push(`<!--]--> <label class="ml-auto flex cursor-pointer items-center gap-1.5 text-muted-foreground select-none"><span>Wrap text</span> `);
              Switch_1($$renderer4, {
                get checked() {
                  return bind_get();
                },
                set checked($$value) {
                  bind_set($$value);
                },
                "aria-label": "Wrap text in the text columns"
              });
              $$renderer4.push(`<!----></label></div> `);
              Hit_table($$renderer4, {
                hits,
                active,
                visible: tableCols,
                widths: tableWidths,
                wrap: tableWrap,
                onwidthchange: setColWidth,
                query: resultsQuery(),
                onselect: (h) => active = h
              });
              $$renderer4.push(`<!---->`);
            } else {
              $$renderer4.push("<!--[-1-->");
              Hit_list($$renderer4, {
                hits,
                query: resultsQuery(),
                active,
                onselect: (h) => active = h
              });
            }
            $$renderer4.push(`<!--]--> `);
            if (!isBrowsing() && hits.length > 0 && !allLoaded) {
              $$renderer4.push("<!--[0-->");
              $$renderer4.push(`<div class="flex justify-center border-t border-border bg-card/40 px-4 py-3">`);
              Button($$renderer4, {
                variant: "outline",
                size: "sm",
                disabled: loadingMore,
                onclick: loadMore,
                children: ($$renderer5) => {
                  $$renderer5.push(`<!---->${escape_html(loadingMore ? "Loading…" : `Show ${PAGE_STEP} more`)}`);
                },
                $$slots: { default: true }
              });
              $$renderer4.push(`<!----></div>`);
            } else {
              $$renderer4.push("<!--[-1-->");
            }
            $$renderer4.push(`<!--]--></div>`);
          }
          $$renderer4.push(`<!--]--></div>`);
        }, right = function($$renderer4) {
          $$renderer4.push(`<div class="h-full min-h-0 bg-muted/30">`);
          Player_pane($$renderer4, { hit: active, query: resultsQuery() });
          $$renderer4.push(`<!----></div>`);
        };
        Resizable_split($$renderer3, {
          minLeft: 420,
          minRight: 360,
          initial: 0.6,
          left,
          right
        });
      }
      $$renderer3.push(`<!----></div>`);
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
