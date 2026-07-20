import "../../../chunks/async.js";
import { s as sanitize_props, a as spread_props, b as slot, m as escape_html, c as setContext, j as bind_props, d as derived, l as ensure_array_like, i as attr_class, f as clsx, h as attr } from "../../../chunks/renderer.js";
import "clsx";
import { tableFromIPC } from "apache-arrow";
import "pixi.js";
import { S as SvelteSet, a as SvelteMap } from "../../../chunks/index-server.js";
import { X, R as Resizable_split } from "../../../chunks/resizable-split.js";
import { B as Button, c as cn } from "../../../chunks/scroll-lock.js";
import { I as Icon } from "../../../chunks/Icon.js";
import { a as Eye, b as Eraser, T as Trash_2, R as Rotate_ccw, E as Eye_off } from "../../../chunks/trash-2.js";
import { M as Minus, I as Input, C as Chevron_left } from "../../../chunks/input.js";
import { C as Check, a as Chevron_down, S as Select_1 } from "../../../chunks/select.js";
import { S as Search } from "../../../chunks/search.js";
import "../../../chunks/descriptor.js";
import { C as Chevron_right } from "../../../chunks/sr-only-styles.js";
function Crosshair($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["circle", { "cx": "12", "cy": "12", "r": "10" }],
    ["line", { "x1": "22", "x2": "18", "y1": "12", "y2": "12" }],
    ["line", { "x1": "6", "x2": "2", "y1": "12", "y2": "12" }],
    ["line", { "x1": "12", "x2": "12", "y1": "6", "y2": "2" }],
    ["line", { "x1": "12", "x2": "12", "y1": "22", "y2": "18" }]
  ];
  Icon($$renderer, spread_props([
    { name: "crosshair" },
    $$sanitized_props,
    {
      /**
       * @component @name Crosshair
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgLz4KICA8bGluZSB4MT0iMjIiIHgyPSIxOCIgeTE9IjEyIiB5Mj0iMTIiIC8+CiAgPGxpbmUgeDE9IjYiIHgyPSIyIiB5MT0iMTIiIHkyPSIxMiIgLz4KICA8bGluZSB4MT0iMTIiIHgyPSIxMiIgeTE9IjYiIHkyPSIyIiAvPgogIDxsaW5lIHgxPSIxMiIgeDI9IjEyIiB5MT0iMjIiIHkyPSIxOCIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/crosshair
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
function Hand($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2" }],
    ["path", { "d": "M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2" }],
    ["path", { "d": "M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8" }],
    [
      "path",
      {
        "d": "M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "hand" },
    $$sanitized_props,
    {
      /**
       * @component @name Hand
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTggMTFWNmEyIDIgMCAwIDAtMi0yYTIgMiAwIDAgMC0yIDIiIC8+CiAgPHBhdGggZD0iTTE0IDEwVjRhMiAyIDAgMCAwLTItMmEyIDIgMCAwIDAtMiAydjIiIC8+CiAgPHBhdGggZD0iTTEwIDEwLjVWNmEyIDIgMCAwIDAtMi0yYTIgMiAwIDAgMC0yIDJ2OCIgLz4KICA8cGF0aCBkPSJNMTggOGEyIDIgMCAxIDEgNCAwdjZhOCA4IDAgMCAxLTggOGgtMmMtMi44IDAtNC41LS44Ni01Ljk5LTIuMzRsLTMuNi0zLjZhMiAyIDAgMCAxIDIuODMtMi44Mkw3IDE1IiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/hand
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
function Lasso($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M7 22a5 5 0 0 1-2-4" }],
    [
      "path",
      {
        "d": "M3.3 14A6.8 6.8 0 0 1 2 10c0-4.4 4.5-8 10-8s10 3.6 10 8-4.5 8-10 8a12 12 0 0 1-5-1"
      }
    ],
    ["path", { "d": "M5 18a2 2 0 1 0 0-4 2 2 0 0 0 0 4z" }]
  ];
  Icon($$renderer, spread_props([
    { name: "lasso" },
    $$sanitized_props,
    {
      /**
       * @component @name Lasso
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNNyAyMmE1IDUgMCAwIDEtMi00IiAvPgogIDxwYXRoIGQ9Ik0zLjMgMTRBNi44IDYuOCAwIDAgMSAyIDEwYzAtNC40IDQuNS04IDEwLThzMTAgMy42IDEwIDgtNC41IDgtMTAgOGExMiAxMiAwIDAgMS01LTEiIC8+CiAgPHBhdGggZD0iTTUgMThhMiAyIDAgMSAwIDAtNCAyIDIgMCAwIDAgMCA0eiIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/lasso
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
function Layers($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"
      }
    ],
    [
      "path",
      {
        "d": "M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"
      }
    ],
    [
      "path",
      {
        "d": "M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "layers" },
    $$sanitized_props,
    {
      /**
       * @component @name Layers
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTIuODMgMi4xOGEyIDIgMCAwIDAtMS42NiAwTDIuNiA2LjA4YTEgMSAwIDAgMCAwIDEuODNsOC41OCAzLjkxYTIgMiAwIDAgMCAxLjY2IDBsOC41OC0zLjlhMSAxIDAgMCAwIDAtMS44M3oiIC8+CiAgPHBhdGggZD0iTTIgMTJhMSAxIDAgMCAwIC41OC45MWw4LjYgMy45MWEyIDIgMCAwIDAgMS42NSAwbDguNTgtMy45QTEgMSAwIDAgMCAyMiAxMiIgLz4KICA8cGF0aCBkPSJNMiAxN2ExIDEgMCAwIDAgLjU4LjkxbDguNiAzLjkxYTIgMiAwIDAgMCAxLjY1IDBsOC41OC0zLjlBMSAxIDAgMCAwIDIyIDE3IiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/layers
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
function Maximize($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M8 3H5a2 2 0 0 0-2 2v3" }],
    ["path", { "d": "M21 8V5a2 2 0 0 0-2-2h-3" }],
    ["path", { "d": "M3 16v3a2 2 0 0 0 2 2h3" }],
    ["path", { "d": "M16 21h3a2 2 0 0 0 2-2v-3" }]
  ];
  Icon($$renderer, spread_props([
    { name: "maximize" },
    $$sanitized_props,
    {
      /**
       * @component @name Maximize
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNOCAzSDVhMiAyIDAgMCAwLTIgMnYzIiAvPgogIDxwYXRoIGQ9Ik0yMSA4VjVhMiAyIDAgMCAwLTItMmgtMyIgLz4KICA8cGF0aCBkPSJNMyAxNnYzYTIgMiAwIDAgMCAyIDJoMyIgLz4KICA8cGF0aCBkPSJNMTYgMjFoM2EyIDIgMCAwIDAgMi0ydi0zIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/maximize
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
function Mouse_pointer_2($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M4.037 4.688a.495.495 0 0 1 .651-.651l16 6.5a.5.5 0 0 1-.063.947l-6.124 1.58a2 2 0 0 0-1.438 1.435l-1.579 6.126a.5.5 0 0 1-.947.063z"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "mouse-pointer-2" },
    $$sanitized_props,
    {
      /**
       * @component @name MousePointer2
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNNC4wMzcgNC42ODhhLjQ5NS40OTUgMCAwIDEgLjY1MS0uNjUxbDE2IDYuNWEuNS41IDAgMCAxLS4wNjMuOTQ3bC02LjEyNCAxLjU4YTIgMiAwIDAgMC0xLjQzOCAxLjQzNWwtMS41NzkgNi4xMjZhLjUuNSAwIDAgMS0uOTQ3LjA2M3oiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/mouse-pointer-2
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
function Paintbrush($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "m14.622 17.897-10.68-2.913" }],
    [
      "path",
      {
        "d": "M18.376 2.622a1 1 0 1 1 3.002 3.002L17.36 9.643a.5.5 0 0 0 0 .707l.944.944a2.41 2.41 0 0 1 0 3.408l-.944.944a.5.5 0 0 1-.707 0L8.354 7.348a.5.5 0 0 1 0-.707l.944-.944a2.41 2.41 0 0 1 3.408 0l.944.944a.5.5 0 0 0 .707 0z"
      }
    ],
    [
      "path",
      {
        "d": "M9 8c-1.804 2.71-3.97 3.46-6.583 3.948a.507.507 0 0 0-.302.819l7.32 8.883a1 1 0 0 0 1.185.204C12.735 20.405 16 16.792 16 15"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "paintbrush" },
    $$sanitized_props,
    {
      /**
       * @component @name Paintbrush
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJtMTQuNjIyIDE3Ljg5Ny0xMC42OC0yLjkxMyIgLz4KICA8cGF0aCBkPSJNMTguMzc2IDIuNjIyYTEgMSAwIDEgMSAzLjAwMiAzLjAwMkwxNy4zNiA5LjY0M2EuNS41IDAgMCAwIDAgLjcwN2wuOTQ0Ljk0NGEyLjQxIDIuNDEgMCAwIDEgMCAzLjQwOGwtLjk0NC45NDRhLjUuNSAwIDAgMS0uNzA3IDBMOC4zNTQgNy4zNDhhLjUuNSAwIDAgMSAwLS43MDdsLjk0NC0uOTQ0YTIuNDEgMi40MSAwIDAgMSAzLjQwOCAwbC45NDQuOTQ0YS41LjUgMCAwIDAgLjcwNyAweiIgLz4KICA8cGF0aCBkPSJNOSA4Yy0xLjgwNCAyLjcxLTMuOTcgMy40Ni02LjU4MyAzLjk0OGEuNTA3LjUwNyAwIDAgMC0uMzAyLjgxOWw3LjMyIDguODgzYTEgMSAwIDAgMCAxLjE4NS4yMDRDMTIuNzM1IDIwLjQwNSAxNiAxNi43OTIgMTYgMTUiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/paintbrush
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
function Pencil($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"
      }
    ],
    ["path", { "d": "m15 5 4 4" }]
  ];
  Icon($$renderer, spread_props([
    { name: "pencil" },
    $$sanitized_props,
    {
      /**
       * @component @name Pencil
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMjEuMTc0IDYuODEyYTEgMSAwIDAgMC0zLjk4Ni0zLjk4N0wzLjg0MiAxNi4xNzRhMiAyIDAgMCAwLS41LjgzbC0xLjMyMSA0LjM1MmEuNS41IDAgMCAwIC42MjMuNjIybDQuMzUzLTEuMzJhMiAyIDAgMCAwIC44My0uNDk3eiIgLz4KICA8cGF0aCBkPSJtMTUgNSA0IDQiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/pencil
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
function Pentagon($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M10.83 2.38a2 2 0 0 1 2.34 0l8 5.74a2 2 0 0 1 .73 2.25l-3.04 9.26a2 2 0 0 1-1.9 1.37H7.04a2 2 0 0 1-1.9-1.37L2.1 10.37a2 2 0 0 1 .73-2.25z"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "pentagon" },
    $$sanitized_props,
    {
      /**
       * @component @name Pentagon
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTAuODMgMi4zOGEyIDIgMCAwIDEgMi4zNCAwbDggNS43NGEyIDIgMCAwIDEgLjczIDIuMjVsLTMuMDQgOS4yNmEyIDIgMCAwIDEtMS45IDEuMzdINy4wNGEyIDIgMCAwIDEtMS45LTEuMzdMMi4xIDEwLjM3YTIgMiAwIDAgMSAuNzMtMi4yNXoiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/pentagon
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
function Spline($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["circle", { "cx": "19", "cy": "5", "r": "2" }],
    ["circle", { "cx": "5", "cy": "19", "r": "2" }],
    ["path", { "d": "M5 17A12 12 0 0 1 17 5" }]
  ];
  Icon($$renderer, spread_props([
    { name: "spline" },
    $$sanitized_props,
    {
      /**
       * @component @name Spline
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjUiIHI9IjIiIC8+CiAgPGNpcmNsZSBjeD0iNSIgY3k9IjE5IiByPSIyIiAvPgogIDxwYXRoIGQ9Ik01IDE3QTEyIDEyIDAgMCAxIDE3IDUiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/spline
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
function Square($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "rect",
      { "width": "18", "height": "18", "x": "3", "y": "3", "rx": "2" }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "square" },
    $$sanitized_props,
    {
      /**
       * @component @name Square
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cmVjdCB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHg9IjMiIHk9IjMiIHJ4PSIyIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/square
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
function Zoom_in($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["circle", { "cx": "11", "cy": "11", "r": "8" }],
    [
      "line",
      { "x1": "21", "x2": "16.65", "y1": "21", "y2": "16.65" }
    ],
    ["line", { "x1": "11", "x2": "11", "y1": "8", "y2": "14" }],
    ["line", { "x1": "8", "x2": "14", "y1": "11", "y2": "11" }]
  ];
  Icon($$renderer, spread_props([
    { name: "zoom-in" },
    $$sanitized_props,
    {
      /**
       * @component @name ZoomIn
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8Y2lyY2xlIGN4PSIxMSIgY3k9IjExIiByPSI4IiAvPgogIDxsaW5lIHgxPSIyMSIgeDI9IjE2LjY1IiB5MT0iMjEiIHkyPSIxNi42NSIgLz4KICA8bGluZSB4MT0iMTEiIHgyPSIxMSIgeTE9IjgiIHkyPSIxNCIgLz4KICA8bGluZSB4MT0iOCIgeDI9IjE0IiB5MT0iMTEiIHkyPSIxMSIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/zoom-in
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
function Zoom_out($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["circle", { "cx": "11", "cy": "11", "r": "8" }],
    [
      "line",
      { "x1": "21", "x2": "16.65", "y1": "21", "y2": "16.65" }
    ],
    ["line", { "x1": "8", "x2": "14", "y1": "11", "y2": "11" }]
  ];
  Icon($$renderer, spread_props([
    { name: "zoom-out" },
    $$sanitized_props,
    {
      /**
       * @component @name ZoomOut
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8Y2lyY2xlIGN4PSIxMSIgY3k9IjExIiByPSI4IiAvPgogIDxsaW5lIHgxPSIyMSIgeDI9IjE2LjY1IiB5MT0iMjEiIHkyPSIxNi42NSIgLz4KICA8bGluZSB4MT0iOCIgeDI9IjE0IiB5MT0iMTEiIHkyPSIxMSIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/zoom-out
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
function AudioViewer($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { unit, onload } = $$props;
    $$renderer2.push(`<div class="flex h-full w-full items-center justify-center p-6 text-center text-sm text-muted-foreground"><div><div class="font-medium">audio annotation — scaffold</div> <div class="mt-1 text-xs">peaks.js waveform · temporal segments (t_start/t_end) · ${escape_html(unit.key)}</div></div></div>`);
  });
}
class LayerStore {
  groupByColumn = "label";
  hiddenGroups = /* @__PURE__ */ new Set();
  groupColors = /* @__PURE__ */ new Map();
  _listeners = /* @__PURE__ */ new Set();
  /** Subscribe to any change. Returns an unsubscribe fn. */
  on(listener) {
    this._listeners.add(listener);
    return () => this._listeners.delete(listener);
  }
  emit() {
    for (const listener of this._listeners) listener();
  }
  toggleVisibility(group) {
    const next = new Set(this.hiddenGroups);
    if (next.has(group)) next.delete(group);
    else next.add(group);
    this.hiddenGroups = next;
    this.emit();
  }
  setColor(group, hex) {
    const next = new Map(this.groupColors);
    next.set(group, hex);
    this.groupColors = next;
    this.emit();
  }
  setGroupBy(column) {
    this.groupByColumn = column;
    this.hiddenGroups = /* @__PURE__ */ new Set();
    this.groupColors = /* @__PURE__ */ new Map();
    this.emit();
  }
  isHidden(group) {
    return this.hiddenGroups.has(group);
  }
  getColor(group) {
    return this.groupColors.get(group);
  }
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
    let { unit, onload, controller } = $$props;
    async function onready(ctx) {
      if (unit.imageUrl) await ctx.plugins.image.load(unit.imageUrl);
      const res = await fetch(unit.annotationsUrl);
      if (!res.ok) throw new Error(`annotations HTTP ${res.status}`);
      const table = tableFromIPC(new Uint8Array(await res.arrayBuffer()));
      ctx.plugins.arrow.load(table);
      ctx.plugins.arrow.sync();
      controller?.attach(ctx, table);
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
const STRING_FIELD_CANDIDATES = ["label", "status", "source", "group", "reviewer"];
function numToHex(n) {
  return "#" + (n & 16777215).toString(16).padStart(6, "0");
}
function hexToNum(hex) {
  return parseInt(hex.replace(/^#/, ""), 16) & 16777215;
}
class AnnotatorController {
  // ── engine handle + data ──
  ctx = null;
  table = null;
  // ── mirrored engine state ──
  mode = "edit";
  activeTool = "select";
  selectedIndex = null;
  selectedSet = new SvelteSet();
  zoomPercent = 1;
  dirty = false;
  count = 0;
  brushOptions = {
    radius: 20,
    erasing: false,
    maskMode: "instance",
    output: "mask"
  };
  // ── layer grouping (mirrored from LayerStore) ──
  layers = new LayerStore();
  groupByColumn = "label";
  hiddenGroups = new SvelteSet();
  groupColors = new SvelteMap();
  // Local field overlay for sidebar/list display, keyed `${index}:${field}` (the
  // canvas is updated separately via arrow.setFieldOverride). SvelteMap ⇒ edits
  // re-derive `rows` with no manual version counter.
  _overrides = new SvelteMap();
  _detachViewport = null;
  constructor() {
    this.layers.on(() => this._pullLayers());
    this._pullLayers();
  }
  #groupColumns = derived(
    // ── derived views the layout reads ──
    /** String columns available to group by. */
    () => {
      const t = this.table;
      if (!t) return [];
      const names = new Set(t.schema.fields.map((f) => f.name));
      return STRING_FIELD_CANDIDATES.filter((c) => names.has(c));
    }
  );
  get groupColumns() {
    return this.#groupColumns();
  }
  set groupColumns($$value) {
    return this.#groupColumns($$value);
  }
  #rows = derived(() => {
    const t = this.table;
    if (!t) return [];
    const out = [];
    for (let i = 0; i < t.numRows; i++) {
      out.push({
        index: i,
        id: this._raw(t, "id", i) ?? String(i),
        label: this._field(t, "label", i) ?? "",
        status: this._field(t, "status", i) ?? "",
        group: this._field(t, "group", i) ?? "",
        text: this._field(t, "text", i) ?? "",
        source: this._field(t, "source", i) ?? "",
        confidence: this._num(t, "confidence", i),
        uncertainty: this._num(t, "uncertainty", i)
      });
    }
    return out;
  });
  get rows() {
    return this.#rows();
  }
  set rows($$value) {
    return this.#rows($$value);
  }
  #groups = derived(() => {
    const col = this.groupByColumn;
    const counts = /* @__PURE__ */ new Map();
    for (const r of this.rows) {
      const key = r[col] ?? "";
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts.entries()].map(([name, count]) => ({ name, count })).sort((a, b) => a.name.localeCompare(b.name));
  });
  get groups() {
    return this.#groups();
  }
  set groups($$value) {
    return this.#groups($$value);
  }
  #selected = derived(() => {
    const i = this.selectedIndex;
    if (i == null) return null;
    return this.rows.find((r) => r.index === i) ?? null;
  });
  get selected() {
    return this.#selected();
  }
  set selected($$value) {
    return this.#selected($$value);
  }
  #canDraw = derived(() => this.mode === "edit");
  get canDraw() {
    return this.#canDraw();
  }
  set canDraw($$value) {
    return this.#canDraw($$value);
  }
  attach(ctx, table) {
    this.ctx = ctx;
    this.table = table;
    this.count = table.numRows;
    const im = ctx.plugins.interaction;
    im.setEditMode(this.mode === "edit");
    im.setTool(this.activeTool);
    im.setBrushOptions(this.brushOptions);
    im.onSelect = (index) => {
      this.selectedIndex = index;
      this._mirrorSelection(im.getSelectedSet());
    };
    im.onDirtyChange = (hasDirty) => {
      if (hasDirty) this.dirty = true;
    };
    im.onCommit = () => {
      this.count = ctx.plugins.arrow.getNumRows();
      this.dirty = true;
    };
    const img = ctx.plugins.image;
    const prev = img.onViewportChange;
    img.onViewportChange = (bounds) => {
      prev?.(bounds);
      this.zoomPercent = img.zoomPercent;
    };
    this._detachViewport = () => {
      img.onViewportChange = prev;
    };
    this.zoomPercent = img.zoomPercent;
    this._syncLayerConfig();
  }
  detach() {
    this._detachViewport?.();
    this._detachViewport = null;
    this.ctx = null;
    this.table = null;
  }
  // ── toolbar / mode ──
  setTool(tool) {
    this.activeTool = tool;
    this.ctx?.plugins.interaction.setTool(tool);
  }
  toggleMode() {
    this.mode = this.mode === "edit" ? "view" : "edit";
    this.ctx?.plugins.interaction.setEditMode(this.mode === "edit");
    if (this.mode === "view") this.setTool("select");
  }
  setBrushOptions(patch) {
    this.brushOptions = { ...this.brushOptions, ...patch };
    this.ctx?.plugins.interaction.setBrushOptions(this.brushOptions);
  }
  // ── selection ──
  select(index) {
    this.selectedIndex = index;
    const im = this.ctx?.plugins.interaction;
    im?.select(index);
    this._mirrorSelection(im?.getSelectedSet() ?? /* @__PURE__ */ new Set());
  }
  deleteSelected() {
    if (this.selectedIndex == null) return;
    this.ctx?.plugins.interaction.handleKeyDown("Delete");
    this.dirty = true;
    this.select(null);
  }
  convertToPolygon() {
    if (this.ctx?.plugins.interaction.convertToPolygon()) this.dirty = true;
  }
  // ── inline field edits (canvas + overlay) ──
  updateField(index, field, value) {
    this._overrides.set(`${index}:${field}`, value);
    this.ctx?.plugins.arrow.setFieldOverride(index, field, value);
    this.ctx?.plugins.arrow.sync();
    this.dirty = true;
  }
  setStatus(index, status) {
    this.updateField(index, "status", status);
  }
  // ── zoom ──
  zoomIn() {
    this.ctx?.plugins.image.zoomIn();
  }
  zoomOut() {
    this.ctx?.plugins.image.zoomOut();
  }
  resetView() {
    this.ctx?.plugins.image.resetView();
  }
  // ── layers ──
  setGroupBy(column) {
    this.layers.setGroupBy(column);
    this.groupByColumn = column;
  }
  toggleGroupVisible(group) {
    this.layers.toggleVisibility(group);
    this.ctx?.plugins.arrow.setGroupVisible(group, !this.layers.isHidden(group));
  }
  setGroupColor(group, hex) {
    this.layers.setColor(group, hexToNum(hex));
  }
  isHidden(group) {
    return this.hiddenGroups.has(group);
  }
  groupColorHex(group) {
    const n = this.groupColors.get(group);
    return n == null ? "#3b82f6" : numToHex(n);
  }
  // ── internals ──
  _mirrorSelection(set) {
    this.selectedSet.clear();
    for (const i of set) this.selectedSet.add(i);
  }
  _pullLayers() {
    this.groupByColumn = this.layers.groupByColumn;
    this.hiddenGroups.clear();
    for (const g of this.layers.hiddenGroups) this.hiddenGroups.add(g);
    this.groupColors.clear();
    for (const [k, v] of this.layers.groupColors) this.groupColors.set(k, v);
    this._syncLayerConfig();
  }
  _syncLayerConfig() {
    const arrow = this.ctx?.plugins.arrow;
    if (!arrow) return;
    arrow.setLayerConfig({
      hiddenGroups: this.layers.hiddenGroups,
      groupByColumn: this.layers.groupByColumn,
      groupColors: this.layers.groupColors
    });
    arrow.sync();
  }
  _raw(t, field, i) {
    const v = t.getChild(field)?.get(i);
    return v == null ? null : String(v);
  }
  _field(t, field, i) {
    const o = this._overrides.get(`${i}:${field}`);
    if (o != null) return o;
    return this._raw(t, field, i);
  }
  _num(t, field, i) {
    const v = t.getChild(field)?.get(i);
    return typeof v === "number" ? v : null;
  }
}
function AnnotatorToolbar($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { controller } = $$props;
    const TOOLS = [
      {
        tool: "select",
        icon: Mouse_pointer_2,
        label: "Select",
        key: "1",
        drawing: false
      },
      {
        tool: "pan",
        icon: Hand,
        label: "Pan",
        key: "2",
        drawing: false
      },
      {
        tool: "rect",
        icon: Square,
        label: "Rectangle",
        key: "3",
        drawing: true
      },
      {
        tool: "polygon",
        icon: Pentagon,
        label: "Polygon",
        key: "4",
        drawing: true
      },
      {
        tool: "point",
        icon: Crosshair,
        label: "Point",
        key: "5",
        drawing: true
      },
      {
        tool: "line",
        icon: Minus,
        label: "Line",
        key: "6",
        drawing: true
      },
      {
        tool: "lasso",
        icon: Lasso,
        label: "Lasso",
        key: "7",
        drawing: true
      },
      {
        tool: "brush",
        icon: Paintbrush,
        label: "Brush",
        key: "B",
        drawing: true
      }
    ];
    const visible = derived(() => TOOLS.filter((t) => !t.drawing || controller.canDraw));
    $$renderer2.push(`<div class="flex h-full w-11 shrink-0 flex-col items-center gap-1 border-r border-border bg-card py-2" data-testid="annotator-toolbar">`);
    Button($$renderer2, {
      variant: controller.mode === "edit" ? "default" : "ghost",
      size: "icon-sm",
      title: controller.mode === "edit" ? "Edit mode (click to view)" : "View mode (click to edit)",
      "aria-pressed": controller.mode === "edit",
      onclick: () => controller.toggleMode(),
      children: ($$renderer3) => {
        if (controller.mode === "edit") {
          $$renderer3.push("<!--[0-->");
          Pencil($$renderer3, { class: "size-4" });
        } else {
          $$renderer3.push("<!--[-1-->");
          Eye($$renderer3, { class: "size-4" });
        }
        $$renderer3.push(`<!--]-->`);
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> <div class="my-1 h-px w-6 bg-border"></div> <!--[-->`);
    const each_array = ensure_array_like(visible());
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let t = each_array[$$index];
      const Icon2 = t.icon;
      Button($$renderer2, {
        variant: controller.activeTool === t.tool ? "default" : "ghost",
        size: "icon-sm",
        title: `${t.label} (${t.key})`,
        "aria-pressed": controller.activeTool === t.tool,
        onclick: () => controller.setTool(t.tool),
        children: ($$renderer3) => {
          if (Icon2) {
            $$renderer3.push("<!--[-->");
            Icon2($$renderer3, { class: "size-4" });
            $$renderer3.push("<!--]-->");
          } else {
            $$renderer3.push("<!--[!-->");
            $$renderer3.push("<!--]-->");
          }
        },
        $$slots: { default: true }
      });
    }
    $$renderer2.push(`<!--]--> `);
    if (controller.activeTool === "brush") {
      $$renderer2.push("<!--[0-->");
      Button($$renderer2, {
        variant: controller.brushOptions.erasing ? "default" : "ghost",
        size: "icon-sm",
        title: "Erase (brush)",
        "aria-pressed": controller.brushOptions.erasing,
        onclick: () => controller.setBrushOptions({ erasing: !controller.brushOptions.erasing }),
        children: ($$renderer3) => {
          Eraser($$renderer3, { class: "size-4" });
        },
        $$slots: { default: true }
      });
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]--> <div class="my-1 h-px w-6 bg-border"></div> `);
    Button($$renderer2, {
      variant: "ghost",
      size: "icon-sm",
      title: "Convert to polygon (P)",
      disabled: controller.selectedIndex == null,
      onclick: () => controller.convertToPolygon(),
      children: ($$renderer3) => {
        Spline($$renderer3, { class: "size-4" });
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> `);
    Button($$renderer2, {
      variant: "ghost",
      size: "icon-sm",
      title: "Delete selected (Del)",
      disabled: controller.selectedIndex == null,
      onclick: () => controller.deleteSelected(),
      children: ($$renderer3) => {
        Trash_2($$renderer3, { class: "size-4" });
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> <div class="mt-auto flex flex-col items-center gap-1"><span${attr_class(clsx(cn("size-2 rounded-full", controller.dirty ? "bg-amber-500" : "bg-transparent")))}${attr("title", controller.dirty ? "Unsaved edits" : "No pending edits")}></span> <span class="text-[10px] tabular-nums text-muted-foreground" title="Annotation count">${escape_html(controller.count)}</span></div></div>`);
  });
}
function statusDot(status) {
  switch (status) {
    case "accepted":
      return "bg-emerald-500";
    case "rejected":
      return "bg-rose-500";
    case "prediction":
      return "bg-amber-500";
    case "reviewed":
      return "bg-sky-500";
    default:
      return "bg-muted-foreground";
  }
}
function statusBadge(status) {
  switch (status) {
    case "accepted":
      return "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400";
    case "rejected":
      return "bg-rose-500/15 text-rose-600 dark:text-rose-400";
    case "prediction":
      return "bg-amber-500/15 text-amber-600 dark:text-amber-400";
    case "reviewed":
      return "bg-sky-500/15 text-sky-600 dark:text-sky-400";
    default:
      return "bg-muted text-muted-foreground";
  }
}
function AnnotationDetail($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { controller } = $$props;
    const row = derived(() => controller.selected);
    if (row()) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="flex flex-col gap-3 p-3" data-testid="annotation-detail"><div class="flex items-center justify-between"><span class="text-xs font-medium text-muted-foreground">Annotation #${escape_html(row().index)}</span> <span${attr_class(clsx(cn("rounded px-1.5 py-0.5 text-[10px] font-medium", statusBadge(row().status))))}>${escape_html(row().status || "—")}</span></div> <label class="flex flex-col gap-1 text-xs"><span class="text-muted-foreground">Text</span> `);
      Input($$renderer2, {
        value: row().text,
        placeholder: "—",
        oninput: (e) => controller.updateField(row().index, "text", e.currentTarget.value)
      });
      $$renderer2.push(`<!----></label> <label class="flex flex-col gap-1 text-xs"><span class="text-muted-foreground">Label</span> `);
      Input($$renderer2, {
        value: row().label,
        placeholder: "—",
        oninput: (e) => controller.updateField(row().index, "label", e.currentTarget.value)
      });
      $$renderer2.push(`<!----></label> <label class="flex flex-col gap-1 text-xs"><span class="text-muted-foreground">Group</span> `);
      Input($$renderer2, {
        value: row().group,
        placeholder: "—",
        oninput: (e) => controller.updateField(row().index, "group", e.currentTarget.value)
      });
      $$renderer2.push(`<!----></label> <dl class="grid grid-cols-2 gap-x-3 gap-y-1 text-xs"><dt class="text-muted-foreground">Source</dt> <dd class="truncate text-right"${attr("title", row().source)}>${escape_html(row().source || "—")}</dd> <dt class="text-muted-foreground">Confidence</dt> <dd class="text-right tabular-nums">${escape_html(row().confidence?.toFixed(2) ?? "—")}</dd> <dt class="text-muted-foreground">Uncertainty</dt> <dd class="text-right tabular-nums">${escape_html(row().uncertainty?.toFixed(2) ?? "—")}</dd></dl> <div class="flex gap-1">`);
      Button($$renderer2, {
        variant: "outline",
        size: "sm",
        class: "flex-1",
        title: "Accept",
        onclick: () => controller.setStatus(row().index, "accepted"),
        children: ($$renderer3) => {
          Check($$renderer3, { class: "size-3.5" });
          $$renderer3.push(`<!----> Accept`);
        },
        $$slots: { default: true }
      });
      $$renderer2.push(`<!----> `);
      Button($$renderer2, {
        variant: "outline",
        size: "sm",
        class: "flex-1",
        title: "Reject",
        onclick: () => controller.setStatus(row().index, "rejected"),
        children: ($$renderer3) => {
          X($$renderer3, { class: "size-3.5" });
          $$renderer3.push(`<!----> Reject`);
        },
        $$slots: { default: true }
      });
      $$renderer2.push(`<!----> `);
      Button($$renderer2, {
        variant: "ghost",
        size: "icon-sm",
        title: "Reset to prediction",
        onclick: () => controller.setStatus(row().index, "prediction"),
        children: ($$renderer3) => {
          Rotate_ccw($$renderer3, { class: "size-3.5" });
        },
        $$slots: { default: true }
      });
      $$renderer2.push(`<!----></div></div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
    }
    $$renderer2.push(`<!--]-->`);
  });
}
function AnnotationList($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { controller } = $$props;
    let filter = "";
    const queue = derived(() => controller.rows.filter((r) => {
      const q = filter.trim().toLowerCase();
      if (!q) return true;
      return (r.text + " " + r.label + " " + r.group).toLowerCase().includes(q);
    }).toSorted((a, b) => {
      const ap = a.status === "prediction" ? 0 : 1;
      const bp = b.status === "prediction" ? 0 : 1;
      if (ap !== bp) return ap - bp;
      return (b.uncertainty ?? -1) - (a.uncertainty ?? -1);
    }));
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="flex min-h-0 flex-1 flex-col" data-testid="annotation-list"><div class="relative px-3 py-2">`);
      Search($$renderer3, {
        class: "pointer-events-none absolute left-5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
      });
      $$renderer3.push(`<!----> `);
      Input($$renderer3, {
        placeholder: "Filter annotations…",
        class: "pl-7",
        get value() {
          return filter;
        },
        set value($$value) {
          filter = $$value;
          $$settled = false;
        }
      });
      $$renderer3.push(`<!----></div> <ul class="min-h-0 flex-1 overflow-y-auto px-1.5 pb-2">`);
      const each_array = ensure_array_like(queue());
      if (each_array.length !== 0) {
        $$renderer3.push("<!--[-->");
        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
          let r = each_array[$$index];
          $$renderer3.push(`<li><button${attr_class(clsx(cn("flex w-full items-start gap-2 rounded px-2 py-1.5 text-left hover:bg-muted/60", controller.selectedIndex === r.index && "bg-primary/10 ring-1 ring-primary/40")))}><span${attr_class(clsx(cn("mt-1 size-2 shrink-0 rounded-full", statusDot(r.status))))}></span> <span class="min-w-0 flex-1"><span class="flex items-center justify-between gap-2"><span class="truncate text-xs font-medium">${escape_html(r.label || `#${r.index}`)}</span> `);
          if (r.uncertainty != null) {
            $$renderer3.push("<!--[0-->");
            $$renderer3.push(`<span class="shrink-0 text-[10px] tabular-nums text-muted-foreground" title="uncertainty">${escape_html(r.uncertainty.toFixed(2))}</span>`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--></span> `);
          if (r.text) {
            $$renderer3.push("<!--[0-->");
            $$renderer3.push(`<span class="block truncate text-[11px] text-muted-foreground">${escape_html(r.text)}</span>`);
          } else {
            $$renderer3.push("<!--[-1-->");
          }
          $$renderer3.push(`<!--]--></span></button></li>`);
        }
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push(`<li class="px-3 py-6 text-center text-xs text-muted-foreground">No annotations</li>`);
      }
      $$renderer3.push(`<!--]--></ul></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}
function LayerPanel($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { controller } = $$props;
    let groupBy = "label";
    const options = derived(() => controller.groupColumns.map((c) => ({ value: c, label: c })));
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<section class="border-t border-border" data-testid="layer-panel"><button class="flex w-full items-center gap-1.5 px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground">`);
      {
        $$renderer3.push("<!--[-1-->");
        Chevron_down($$renderer3, { class: "size-3.5" });
      }
      $$renderer3.push(`<!--]--> `);
      Layers($$renderer3, { class: "size-3.5" });
      $$renderer3.push(`<!----> Layers</button> `);
      {
        $$renderer3.push("<!--[0-->");
        $$renderer3.push(`<div class="flex flex-col gap-2 px-3 pb-3"><label class="flex items-center gap-2 text-xs text-muted-foreground"><span class="shrink-0">Group by</span> `);
        Select_1($$renderer3, {
          options: options(),
          get value() {
            return groupBy;
          },
          set value($$value) {
            groupBy = $$value;
            $$settled = false;
          }
        });
        $$renderer3.push(`<!----></label> <ul class="flex flex-col gap-0.5"><!--[-->`);
        const each_array = ensure_array_like(controller.groups);
        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
          let g = each_array[$$index];
          $$renderer3.push(`<li class="flex items-center gap-2 rounded px-1 py-0.5 hover:bg-muted/50"><button class="text-muted-foreground hover:text-foreground"${attr("title", controller.isHidden(g.name) ? "Show" : "Hide")}>`);
          if (controller.isHidden(g.name)) {
            $$renderer3.push("<!--[0-->");
            Eye_off($$renderer3, { class: "size-3.5" });
          } else {
            $$renderer3.push("<!--[-1-->");
            Eye($$renderer3, { class: "size-3.5" });
          }
          $$renderer3.push(`<!--]--></button> <input type="color" class="size-3.5 shrink-0 cursor-pointer rounded border-0 bg-transparent p-0"${attr("value", controller.groupColorHex(g.name))} title="Group color"/> <span${attr_class(clsx(cn("flex-1 truncate text-xs", controller.isHidden(g.name) && "text-muted-foreground line-through")))}${attr("title", g.name)}>${escape_html(g.name || "∅")}</span> <span class="text-[10px] tabular-nums text-muted-foreground">${escape_html(g.count)}</span></li>`);
        }
        $$renderer3.push(`<!--]--></ul></div>`);
      }
      $$renderer3.push(`<!--]--></section>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}
function AnnotationSidebar($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { controller } = $$props;
    const summary = derived(() => {
      const counts = /* @__PURE__ */ new Map();
      for (const r of controller.rows) counts.set(r.status, (counts.get(r.status) ?? 0) + 1);
      return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    });
    $$renderer2.push(`<aside class="flex h-full w-full min-w-0 flex-col border-l border-border bg-card" data-testid="annotation-sidebar"><header class="flex items-center justify-between border-b border-border px-3 py-2"><h2 class="text-sm font-semibold">Review queue</h2> <span class="text-xs tabular-nums text-muted-foreground">${escape_html(controller.count)}</span></header> <div class="flex flex-wrap gap-1.5 border-b border-border px-3 py-2"><!--[-->`);
    const each_array = ensure_array_like(summary());
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let [status, n] = each_array[$$index];
      $$renderer2.push(`<span class="flex items-center gap-1 text-[11px] text-muted-foreground"><span${attr_class(clsx(cn("size-2 rounded-full", statusDot(status))))}></span> ${escape_html(status || "—")} <span class="tabular-nums">${escape_html(n)}</span></span>`);
    }
    $$renderer2.push(`<!--]--></div> <div class="flex min-h-0 flex-1 flex-col">`);
    if (controller.selected) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div class="border-b border-border px-2 py-1.5">`);
      Button($$renderer2, {
        variant: "ghost",
        size: "sm",
        onclick: () => controller.select(null),
        children: ($$renderer3) => {
          Chevron_left($$renderer3, { class: "size-3.5" });
          $$renderer3.push(`<!----> Back to list`);
        },
        $$slots: { default: true }
      });
      $$renderer2.push(`<!----></div> <div class="min-h-0 flex-1 overflow-y-auto">`);
      AnnotationDetail($$renderer2, { controller });
      $$renderer2.push(`<!----></div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
      AnnotationList($$renderer2, { controller });
    }
    $$renderer2.push(`<!--]--></div> `);
    LayerPanel($$renderer2, { controller });
    $$renderer2.push(`<!----></aside>`);
  });
}
function ZoomControls($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { controller } = $$props;
    const pct = derived(() => Math.round(controller.zoomPercent * 100));
    $$renderer2.push(`<div class="pointer-events-auto absolute bottom-2 right-2 z-10 flex items-center gap-0.5 rounded-lg border border-border bg-card/90 p-0.5 shadow-md backdrop-blur" data-testid="zoom-controls">`);
    Button($$renderer2, {
      variant: "ghost",
      size: "icon-xs",
      title: "Zoom out",
      onclick: () => controller.zoomOut(),
      children: ($$renderer3) => {
        Zoom_out($$renderer3, { class: "size-4" });
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> <button class="min-w-11 rounded px-1 py-0.5 text-center text-xs tabular-nums text-muted-foreground hover:text-foreground" title="Reset to fit">${escape_html(pct())}%</button> `);
    Button($$renderer2, {
      variant: "ghost",
      size: "icon-xs",
      title: "Zoom in",
      onclick: () => controller.zoomIn(),
      children: ($$renderer3) => {
        Zoom_in($$renderer3, { class: "size-4" });
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> <div class="mx-0.5 h-4 w-px bg-border"></div> `);
    Button($$renderer2, {
      variant: "ghost",
      size: "icon-xs",
      title: "Fit to view",
      onclick: () => controller.resetView(),
      children: ($$renderer3) => {
        Maximize($$renderer3, { class: "size-4" });
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----></div>`);
  });
}
function PageNav($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { pages, current = 0, onNavigate } = $$props;
    const total = derived(() => pages.length);
    const go = (i) => {
      if (i >= 0 && i < total()) onNavigate?.(i);
    };
    $$renderer2.push(`<nav class="pointer-events-auto absolute bottom-2 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1 rounded-lg border border-border bg-card/90 px-1 py-0.5 shadow-md backdrop-blur" data-testid="page-nav">`);
    Button($$renderer2, {
      variant: "ghost",
      size: "icon-xs",
      title: "Previous page",
      disabled: current <= 0,
      onclick: () => go(current - 1),
      children: ($$renderer3) => {
        Chevron_left($$renderer3, { class: "size-4" });
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----> <span class="px-1 text-xs tabular-nums text-muted-foreground">${escape_html(total() === 0 ? "0 / 0" : `${current + 1} / ${total()}`)}</span> `);
    Button($$renderer2, {
      variant: "ghost",
      size: "icon-xs",
      title: "Next page",
      disabled: current >= total() - 1,
      onclick: () => go(current + 1),
      children: ($$renderer3) => {
        Chevron_right($$renderer3, { class: "size-4" });
      },
      $$slots: { default: true }
    });
    $$renderer2.push(`<!----></nav>`);
  });
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
    const controller = new AnnotatorController();
    let status = "loading…";
    const pages = [{ key: KEY, label: "p19" }];
    let pageIndex = 0;
    $$renderer2.push(`<div class="flex h-screen w-screen">`);
    AnnotatorToolbar($$renderer2, { controller });
    $$renderer2.push(`<!----> <div class="min-w-0 flex-1">`);
    {
      let left = function($$renderer3) {
        $$renderer3.push(`<div class="relative h-full w-full"><div class="absolute left-3 top-3 z-10 rounded bg-black/70 px-2 py-1 font-mono text-xs text-white" data-testid="annotate-status">annotate · ${escape_html(unit.kind)} · ${escape_html(status)}</div> `);
        Viewer($$renderer3, {
          unit,
          controller,
          onload: (n) => status = `${n} annotations from Lance`
        });
        $$renderer3.push(`<!----> `);
        PageNav($$renderer3, { pages, current: pageIndex, onNavigate: (i) => pageIndex = i });
        $$renderer3.push(`<!----> `);
        ZoomControls($$renderer3, { controller });
        $$renderer3.push(`<!----></div>`);
      }, right = function($$renderer3) {
        AnnotationSidebar($$renderer3, { controller });
      };
      Resizable_split($$renderer2, {
        storageKey: "lance-media-annotate",
        initial: 0.72,
        minLeft: 420,
        minRight: 320,
        left,
        right
      });
    }
    $$renderer2.push(`<!----></div></div>`);
  });
}
export {
  _page as default
};
