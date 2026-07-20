import "../../chunks/async.js";
import { s as sanitize_props, a as spread_props, b as slot, c as setContext, g as getContext, d as derived, e as attributes, f as clsx, h as attr, i as attr_class, j as bind_props, k as stringify, l as ensure_array_like, m as escape_html } from "../../chunks/renderer.js";
import { p as page } from "../../chunks/index3.js";
import "clsx";
import { g as getHealth, a as getDatasetView, s as setActiveView } from "../../chunks/api.js";
import { c as cn, B as Button, P as Portal } from "../../chunks/scroll-lock.js";
import { I as Icon } from "../../chunks/Icon.js";
import { tv } from "tailwind-variants";
import { P as Popover, a as Popover_trigger, b as Popover_content, A as Audio_lines } from "../../chunks/popover.js";
import { S as Search } from "../../chunks/search.js";
import { M as Map } from "../../chunks/map.js";
class DescriptorStore {
  view = null;
  error = null;
  /** The dataset id from `?dataset=`, or null for the default DB. */
  paramId() {
    if (typeof location === "undefined") return null;
    return new URLSearchParams(location.search).get("dataset");
  }
  async load() {
    try {
      const param = this.paramId();
      let id;
      let isDefault;
      if (param) {
        id = param;
        isDefault = false;
      } else {
        const health = await getHealth();
        id = health.db.path.replace(/\.lance$/, "").split("/").pop() ?? "";
        isDefault = true;
      }
      const view = await getDatasetView(id, isDefault);
      setActiveView(view);
      this.view = view;
      if (typeof window !== "undefined") {
        window.__activeDataset = {
          id: view.id,
          keyFields: view.keyFields,
          hasTime: view.hasTime
        };
      }
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    }
  }
}
const descriptor = new DescriptorStore();
function Activity($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "activity" },
    $$sanitized_props,
    {
      /**
       * @component @name Activity
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMjIgMTJoLTIuNDhhMiAyIDAgMCAwLTEuOTMgMS40NmwtMi4zNSA4LjM2YS4yNS4yNSAwIDAgMS0uNDggMEw5LjI0IDIuMThhLjI1LjI1IDAgMCAwLS40OCAwbC0yLjM1IDguMzZBMiAyIDAgMCAxIDQuNDkgMTJIMiIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/activity
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
function Book_open($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["path", { "d": "M12 7v14" }],
    [
      "path",
      {
        "d": "M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "book-open" },
    $$sanitized_props,
    {
      /**
       * @component @name BookOpen
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTIgN3YxNCIgLz4KICA8cGF0aCBkPSJNMyAxOGExIDEgMCAwIDEtMS0xVjRhMSAxIDAgMCAxIDEtMWg1YTQgNCAwIDAgMSA0IDQgNCA0IDAgMCAxIDQtNGg1YTEgMSAwIDAgMSAxIDF2MTNhMSAxIDAgMCAxLTEgMWgtNmEzIDMgMCAwIDAtMyAzIDMgMyAwIDAgMC0zLTN6IiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/book-open
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
function Folder_tree($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M20 10a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1h-2.5a1 1 0 0 1-.8-.4l-.9-1.2A1 1 0 0 0 15 3h-2a1 1 0 0 0-1 1v5a1 1 0 0 0 1 1Z"
      }
    ],
    [
      "path",
      {
        "d": "M20 21a1 1 0 0 0 1-1v-3a1 1 0 0 0-1-1h-2.9a1 1 0 0 1-.88-.55l-.42-.85a1 1 0 0 0-.92-.6H13a1 1 0 0 0-1 1v5a1 1 0 0 0 1 1Z"
      }
    ],
    ["path", { "d": "M3 5a2 2 0 0 0 2 2h3" }],
    ["path", { "d": "M3 3v13a2 2 0 0 0 2 2h3" }]
  ];
  Icon($$renderer, spread_props([
    { name: "folder-tree" },
    $$sanitized_props,
    {
      /**
       * @component @name FolderTree
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMjAgMTBhMSAxIDAgMCAwIDEtMVY2YTEgMSAwIDAgMC0xLTFoLTIuNWExIDEgMCAwIDEtLjgtLjRsLS45LTEuMkExIDEgMCAwIDAgMTUgM2gtMmExIDEgMCAwIDAtMSAxdjVhMSAxIDAgMCAwIDEgMVoiIC8+CiAgPHBhdGggZD0iTTIwIDIxYTEgMSAwIDAgMCAxLTF2LTNhMSAxIDAgMCAwLTEtMWgtMi45YTEgMSAwIDAgMS0uODgtLjU1bC0uNDItLjg1YTEgMSAwIDAgMC0uOTItLjZIMTNhMSAxIDAgMCAwLTEgMXY1YTEgMSAwIDAgMCAxIDFaIiAvPgogIDxwYXRoIGQ9Ik0zIDVhMiAyIDAgMCAwIDIgMmgzIiAvPgogIDxwYXRoIGQ9Ik0zIDN2MTNhMiAyIDAgMCAwIDIgMmgzIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/folder-tree
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
function Moon($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [["path", { "d": "M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" }]];
  Icon($$renderer, spread_props([
    { name: "moon" },
    $$sanitized_props,
    {
      /**
       * @component @name Moon
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTIgM2E2IDYgMCAwIDAgOSA5IDkgOSAwIDEgMS05LTlaIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/moon
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
function Panel_left($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "rect",
      { "width": "18", "height": "18", "x": "3", "y": "3", "rx": "2" }
    ],
    ["path", { "d": "M9 3v18" }]
  ];
  Icon($$renderer, spread_props([
    { name: "panel-left" },
    $$sanitized_props,
    {
      /**
       * @component @name PanelLeft
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cmVjdCB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHg9IjMiIHk9IjMiIHJ4PSIyIiAvPgogIDxwYXRoIGQ9Ik05IDN2MTgiIC8+Cjwvc3ZnPgo=) - https://lucide.dev/icons/panel-left
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
function Share_2($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    ["circle", { "cx": "18", "cy": "5", "r": "3" }],
    ["circle", { "cx": "6", "cy": "12", "r": "3" }],
    ["circle", { "cx": "18", "cy": "19", "r": "3" }],
    [
      "line",
      { "x1": "8.59", "x2": "15.42", "y1": "13.51", "y2": "17.49" }
    ],
    [
      "line",
      { "x1": "15.41", "x2": "8.59", "y1": "6.51", "y2": "10.49" }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "share-2" },
    $$sanitized_props,
    {
      /**
       * @component @name Share2
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8Y2lyY2xlIGN4PSIxOCIgY3k9IjUiIHI9IjMiIC8+CiAgPGNpcmNsZSBjeD0iNiIgY3k9IjEyIiByPSIzIiAvPgogIDxjaXJjbGUgY3g9IjE4IiBjeT0iMTkiIHI9IjMiIC8+CiAgPGxpbmUgeDE9IjguNTkiIHgyPSIxNS40MiIgeTE9IjEzLjUxIiB5Mj0iMTcuNDkiIC8+CiAgPGxpbmUgeDE9IjE1LjQxIiB4Mj0iOC41OSIgeTE9IjYuNTEiIHkyPSIxMC40OSIgLz4KPC9zdmc+Cg==) - https://lucide.dev/icons/share-2
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
function Square_pen($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "path",
      {
        "d": "M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"
      }
    ],
    [
      "path",
      {
        "d": "M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z"
      }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "square-pen" },
    $$sanitized_props,
    {
      /**
       * @component @name SquarePen
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTIgM0g1YTIgMiAwIDAgMC0yIDJ2MTRhMiAyIDAgMCAwIDIgMmgxNGEyIDIgMCAwIDAgMi0ydi03IiAvPgogIDxwYXRoIGQ9Ik0xOC4zNzUgMi42MjVhMSAxIDAgMCAxIDMgM2wtOS4wMTMgOS4wMTRhMiAyIDAgMCAxLS44NTMuNTA1bC0yLjg3My44NGEuNS41IDAgMCAxLS42Mi0uNjJsLjg0LTIuODczYTIgMiAwIDAgMSAuNTA2LS44NTJ6IiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/square-pen
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
function Workflow($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const iconNode = [
    [
      "rect",
      { "width": "8", "height": "8", "x": "3", "y": "3", "rx": "2" }
    ],
    ["path", { "d": "M7 11v4a2 2 0 0 0 2 2h4" }],
    [
      "rect",
      { "width": "8", "height": "8", "x": "13", "y": "13", "rx": "2" }
    ]
  ];
  Icon($$renderer, spread_props([
    { name: "workflow" },
    $$sanitized_props,
    {
      /**
       * @component @name Workflow
       * @description Lucide SVG icon component, renders SVG Element with children.
       *
       * @preview ![img](data:image/svg+xml;base64,PHN2ZyAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogIHdpZHRoPSIyNCIKICBoZWlnaHQ9IjI0IgogIHZpZXdCb3g9IjAgMCAyNCAyNCIKICBmaWxsPSJub25lIgogIHN0cm9rZT0iIzAwMCIgc3R5bGU9ImJhY2tncm91bmQtY29sb3I6ICNmZmY7IGJvcmRlci1yYWRpdXM6IDJweCIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9pbj0icm91bmQiCj4KICA8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0PSI4IiB4PSIzIiB5PSIzIiByeD0iMiIgLz4KICA8cGF0aCBkPSJNNyAxMXY0YTIgMiAwIDAgMCAyIDJoNCIgLz4KICA8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0PSI4IiB4PSIxMyIgeT0iMTMiIHJ4PSIyIiAvPgo8L3N2Zz4K) - https://lucide.dev/icons/workflow
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
class SidebarState {
  props;
  #open = derived(() => this.props.open());
  get open() {
    return this.#open();
  }
  set open($$value) {
    return this.#open($$value);
  }
  setOpen;
  #state = derived(() => this.open ? "expanded" : "collapsed");
  get state() {
    return this.#state();
  }
  set state($$value) {
    return this.#state($$value);
  }
  constructor(props) {
    this.setOpen = props.setOpen;
    this.props = props;
  }
  toggle = () => this.setOpen(!this.open);
}
const SYMBOL_KEY = "scn-sidebar";
function setSidebar(props) {
  return setContext(Symbol.for(SYMBOL_KEY), new SidebarState(props));
}
function useSidebar() {
  return getContext(Symbol.for(SYMBOL_KEY));
}
function Sidebar($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      ref = null,
      side = "left",
      collapsible = "icon",
      class: className,
      children,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const sidebar = useSidebar();
    if (collapsible === "none") {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<div${attributes({
        "data-slot": "sidebar",
        class: clsx(cn("bg-sidebar text-sidebar-foreground flex h-full w-(--sidebar-width) flex-col", className)),
        ...restProps
      })}>`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></div>`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div class="group peer text-sidebar-foreground block" data-slot="sidebar"${attr("data-state", sidebar.state)}${attr("data-collapsible", sidebar.state === "collapsed" ? collapsible : "")}${attr("data-side", side)}><div data-slot="sidebar-gap"${attr_class(clsx(cn("relative w-(--sidebar-width) bg-transparent transition-[width] duration-200 ease-linear", "group-data-[collapsible=offcanvas]:w-0", "group-data-[side=right]:rotate-180", "group-data-[collapsible=icon]:w-(--sidebar-width-icon)")))}></div> <div${attributes({
        "data-slot": "sidebar-container",
        class: clsx(cn(
          "fixed inset-y-0 z-10 flex h-svh w-(--sidebar-width) transition-[left,right,width] duration-200 ease-linear",
          side === "left" ? "left-0 group-data-[collapsible=offcanvas]:left-[calc(var(--sidebar-width)*-1)]" : "right-0 group-data-[collapsible=offcanvas]:right-[calc(var(--sidebar-width)*-1)]",
          "group-data-[collapsible=icon]:w-(--sidebar-width-icon)",
          "group-data-[side=left]:border-r group-data-[side=right]:border-l border-sidebar-border",
          className
        )),
        ...restProps
      })}><div data-sidebar="sidebar" data-slot="sidebar-inner" class="bg-sidebar flex h-full w-full flex-col">`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></div></div></div>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
const SIDEBAR_COOKIE_NAME = "sidebar:state";
const SIDEBAR_COOKIE_MAX_AGE = 60 * 60 * 24 * 7;
const SIDEBAR_WIDTH = "15rem";
const SIDEBAR_WIDTH_ICON = "3.25rem";
function Sidebar_provider($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      ref = null,
      open = true,
      onOpenChange = () => {
      },
      class: className,
      style,
      children,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    setSidebar({
      open: () => open,
      setOpen: (value) => {
        open = value;
        onOpenChange(value);
        document.cookie = `${SIDEBAR_COOKIE_NAME}=${open}; path=/; max-age=${SIDEBAR_COOKIE_MAX_AGE}`;
      }
    });
    $$renderer2.push(`<div${attributes({
      "data-slot": "sidebar-wrapper",
      style: `--sidebar-width: ${stringify(SIDEBAR_WIDTH)}; --sidebar-width-icon: ${stringify(SIDEBAR_WIDTH_ICON)}; ${stringify(style ?? "")}`,
      class: clsx(cn("group/sidebar-wrapper text-sidebar-foreground flex min-h-svh w-full", className)),
      ...restProps
    })}>`);
    children?.($$renderer2);
    $$renderer2.push(`<!----></div>`);
    bind_props($$props, { ref, open });
  });
}
function Sidebar_trigger($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      ref = null,
      class: className,
      onclick,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    const sidebar = useSidebar();
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Button($$renderer3, spread_props([
        {
          "data-sidebar": "trigger",
          "data-slot": "sidebar-trigger",
          variant: "ghost",
          size: "icon",
          class: cn("size-8", className),
          title: "Toggle sidebar (⌘/Ctrl-B)",
          "aria-label": "Toggle sidebar",
          onclick: (e) => {
            onclick?.(e);
            sidebar.toggle();
          }
        },
        restProps,
        {
          get ref() {
            return ref;
          },
          set ref($$value) {
            ref = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            Panel_left($$renderer4, {});
            $$renderer4.push(`<!----> <span class="sr-only">Toggle sidebar</span>`);
          },
          $$slots: { default: true }
        }
      ]));
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { ref });
  });
}
function Sidebar_rail($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      ref = null,
      class: className,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    useSidebar();
    $$renderer2.push(`<button${attributes({
      "data-slot": "sidebar-rail",
      "data-sidebar": "rail",
      "aria-label": "Toggle sidebar",
      tabindex: -1,
      title: "Toggle sidebar",
      class: clsx(cn("absolute inset-y-0 z-20 hidden w-4 -translate-x-1/2 transition-all ease-linear after:absolute after:inset-y-0 after:left-1/2 after:w-[2px] hover:after:bg-sidebar-border sm:flex", "group-data-[side=left]:-right-4 group-data-[side=right]:left-0", "group-data-[side=left]:cursor-w-resize group-data-[side=right]:cursor-e-resize", "group-data-[collapsible=icon]:cursor-e-resize", className)),
      ...restProps
    })}></button>`);
    bind_props($$props, { ref });
  });
}
function Sidebar_inset($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      ref = null,
      class: className,
      children,
      $$slots,
      $$events,
      ...restProps
    } = $$props;
    $$renderer2.push(`<main${attributes({
      "data-slot": "sidebar-inset",
      class: clsx(cn("bg-background relative flex min-h-0 min-w-0 flex-1 flex-col", className)),
      ...restProps
    })}>`);
    children?.($$renderer2);
    $$renderer2.push(`<!----></main>`);
    bind_props($$props, { ref });
  });
}
const sidebarMenuButtonVariants = tv({
  base: "peer/menu-button ring-sidebar-ring hover:bg-sidebar-accent hover:text-sidebar-accent-foreground active:bg-sidebar-accent active:text-sidebar-accent-foreground data-[active=true]:bg-sidebar-accent data-[active=true]:text-sidebar-accent-foreground data-[active=true]:font-medium flex w-full items-center gap-2 overflow-hidden rounded-md p-2 text-left text-sm outline-hidden transition-[width,height,padding] focus-visible:ring-2 disabled:pointer-events-none disabled:opacity-50 group-data-[collapsible=icon]:size-8! group-data-[collapsible=icon]:p-2! [&>span:last-child]:truncate [&>svg]:size-4 [&>svg]:shrink-0",
  variants: {
    variant: {
      default: "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
      outline: "bg-background shadow-[0_0_0_1px_var(--sidebar-border)] hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
    },
    size: {
      default: "h-9 text-sm",
      sm: "h-7 text-xs",
      lg: "h-12 text-sm group-data-[collapsible=icon]:p-0!"
    }
  },
  defaultVariants: { variant: "default", size: "default" }
});
function Sidebar_menu_button($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      ref = null,
      class: className,
      variant = "default",
      size = "default",
      isActive = false,
      tooltip,
      href = void 0,
      children,
      $$slots,
      $$events,
      ...restProps
      /** native tooltip — useful when the rail is collapsed to icons */
    } = $$props;
    if (href) {
      $$renderer2.push("<!--[0-->");
      $$renderer2.push(`<a${attributes({
        "data-slot": "sidebar-menu-button",
        "data-sidebar": "menu-button",
        "data-size": size,
        "data-active": isActive,
        title: tooltip,
        href,
        class: clsx(cn(sidebarMenuButtonVariants({ variant, size }), className)),
        ...restProps
      })}>`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></a>`);
    } else {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<button${attributes({
        "data-slot": "sidebar-menu-button",
        "data-sidebar": "menu-button",
        "data-size": size,
        "data-active": isActive,
        title: tooltip,
        class: clsx(cn(sidebarMenuButtonVariants({ variant, size }), className)),
        ...restProps
      })}>`);
      children?.($$renderer2);
      $$renderer2.push(`<!----></button>`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { ref });
  });
}
function Theme_toggle($$renderer) {
  function toggle() {
    return;
  }
  Button($$renderer, {
    variant: "ghost",
    size: "icon",
    title: "Switch to dark",
    "aria-label": "Toggle theme",
    onclick: toggle,
    children: ($$renderer2) => {
      {
        $$renderer2.push("<!--[-1-->");
        Moon($$renderer2, { class: "size-4" });
      }
      $$renderer2.push(`<!--]-->`);
    },
    $$slots: { default: true }
  });
}
function Status_badge($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const tone = derived(() => {
      return "red";
    });
    if (Popover) {
      $$renderer2.push("<!--[-->");
      Popover($$renderer2, {
        children: ($$renderer3) => {
          if (Popover_trigger) {
            $$renderer3.push("<!--[-->");
            Popover_trigger($$renderer3, {
              class: "flex items-center gap-1.5 rounded px-2 py-1 text-[11px] hover:bg-secondary/60",
              title: "Service status",
              children: ($$renderer4) => {
                $$renderer4.push(`<span${attr_class(`size-2 rounded-full ${stringify(tone() === "green" ? "bg-emerald-500" : tone() === "amber" ? "bg-amber-500" : "bg-red-500")}`)}></span> `);
                Activity($$renderer4, { class: "size-3.5 text-muted-foreground" });
                $$renderer4.push(`<!----> <span class="hidden text-muted-foreground sm:inline group-data-[collapsible=icon]:!hidden">services</span>`);
              },
              $$slots: { default: true }
            });
            $$renderer3.push("<!--]-->");
          } else {
            $$renderer3.push("<!--[!-->");
            $$renderer3.push("<!--]-->");
          }
          $$renderer3.push(` `);
          if (Portal) {
            $$renderer3.push("<!--[-->");
            Portal($$renderer3, {
              children: ($$renderer4) => {
                if (Popover_content) {
                  $$renderer4.push("<!--[-->");
                  Popover_content($$renderer4, {
                    side: "top",
                    sideOffset: 8,
                    align: "start",
                    class: "z-50 w-80 rounded-md border border-border bg-card p-3 text-xs shadow-md",
                    children: ($$renderer5) => {
                      {
                        $$renderer5.push("<!--[1-->");
                        $$renderer5.push(`<div class="text-muted-foreground">Loading…</div>`);
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
              }
            });
            $$renderer3.push("<!--]-->");
          } else {
            $$renderer3.push("<!--[!-->");
            $$renderer3.push("<!--]-->");
          }
        },
        $$slots: { default: true }
      });
      $$renderer2.push("<!--]-->");
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push("<!--]-->");
    }
  });
}
function _layout($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { children } = $$props;
    const NAV = [
      {
        href: "/",
        label: "Search",
        icon: Search,
        hint: "Find moments across transcripts, video & scenes"
      },
      {
        href: "/atlas",
        label: "Atlas",
        icon: Map,
        hint: "Explore the embedding map of every chunk"
      },
      {
        href: "/tree",
        label: "Tree",
        icon: Folder_tree,
        hint: "Browse topics as a zoomable treemap"
      },
      {
        href: "/graph",
        label: "Graph",
        icon: Share_2,
        hint: "Explore the knowledge graph — entities, relations & clips"
      },
      {
        href: "/workflow",
        label: "Workflow",
        icon: Workflow,
        hint: "Build & run a retrieval pipeline as a node graph"
      },
      {
        href: "/annotate",
        label: "Annotate",
        icon: Square_pen,
        hint: "View & annotate documents/HTR — canvas + review-queue table"
      },
      {
        href: "/guide",
        label: "Guide",
        icon: Book_open,
        hint: "How search works — signals, fusion & rerank"
      }
    ];
    const isActive = (href) => href === "/" ? page.url.pathname === "/" : page.url.pathname.startsWith(href);
    const current = derived(() => NAV.find((n) => isActive(n.href)));
    function initialOpen() {
      return true;
    }
    let sidebarOpen = initialOpen();
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (Sidebar_provider) {
        $$renderer3.push("<!--[-->");
        Sidebar_provider($$renderer3, {
          class: "h-svh overflow-hidden",
          get open() {
            return sidebarOpen;
          },
          set open($$value) {
            sidebarOpen = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            if (Sidebar) {
              $$renderer4.push("<!--[-->");
              Sidebar($$renderer4, {
                collapsible: "icon",
                children: ($$renderer5) => {
                  $$renderer5.push(`<div data-slot="sidebar-header" class="flex flex-col gap-2 p-2"><a href="/" title="lance-media — home" aria-label="lance-media home" class="hover:bg-sidebar-accent flex items-center gap-2 rounded-md p-1.5 transition-colors"><div class="bg-primary/10 text-primary grid size-8 shrink-0 place-items-center rounded-lg">`);
                  Audio_lines($$renderer5, { class: "size-5" });
                  $$renderer5.push(`<!----></div> <div class="flex min-w-0 flex-col leading-tight group-data-[collapsible=icon]:hidden"><span class="text-foreground truncate text-sm font-semibold">lance-media</span> <span class="text-muted-foreground truncate text-[10px]">press-conf viewer</span></div></a></div> <div data-slot="sidebar-content" class="flex min-h-0 flex-1 flex-col gap-2 overflow-auto group-data-[collapsible=icon]:overflow-hidden"><div data-slot="sidebar-group" class="relative flex w-full min-w-0 flex-col p-2"><div data-slot="sidebar-group-label" class="text-sidebar-foreground/60 flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium tracking-wide uppercase transition-[margin,opacity] duration-200 ease-linear group-data-[collapsible=icon]:-mt-8 group-data-[collapsible=icon]:opacity-0">Navigate</div> <ul data-slot="sidebar-menu" class="flex w-full min-w-0 flex-col gap-1"><!--[-->`);
                  const each_array = ensure_array_like(NAV);
                  for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
                    let item = each_array[$$index];
                    const active = isActive(item.href);
                    const Icon2 = item.icon;
                    $$renderer5.push(`<li data-slot="sidebar-menu-item" class="group/menu-item relative">`);
                    if (Sidebar_menu_button) {
                      $$renderer5.push("<!--[-->");
                      Sidebar_menu_button($$renderer5, {
                        href: item.href,
                        isActive: active,
                        tooltip: item.label,
                        children: ($$renderer6) => {
                          if (Icon2) {
                            $$renderer6.push("<!--[-->");
                            Icon2($$renderer6, {});
                            $$renderer6.push("<!--]-->");
                          } else {
                            $$renderer6.push("<!--[!-->");
                            $$renderer6.push("<!--]-->");
                          }
                          $$renderer6.push(` <span>${escape_html(item.label)}</span>`);
                        },
                        $$slots: { default: true }
                      });
                      $$renderer5.push("<!--]-->");
                    } else {
                      $$renderer5.push("<!--[!-->");
                      $$renderer5.push("<!--]-->");
                    }
                    $$renderer5.push(`</li>`);
                  }
                  $$renderer5.push(`<!--]--></ul></div></div> <div data-slot="sidebar-footer" class="flex flex-col gap-2 p-2"><div class="flex items-center justify-between gap-1 group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:gap-2">`);
                  Status_badge($$renderer5);
                  $$renderer5.push(`<!----> `);
                  Theme_toggle($$renderer5);
                  $$renderer5.push(`<!----></div></div> `);
                  if (Sidebar_rail) {
                    $$renderer5.push("<!--[-->");
                    Sidebar_rail($$renderer5, {});
                    $$renderer5.push("<!--]-->");
                  } else {
                    $$renderer5.push("<!--[!-->");
                    $$renderer5.push("<!--]-->");
                  }
                },
                $$slots: { default: true }
              });
              $$renderer4.push("<!--]-->");
            } else {
              $$renderer4.push("<!--[!-->");
              $$renderer4.push("<!--]-->");
            }
            $$renderer4.push(` `);
            if (Sidebar_inset) {
              $$renderer4.push("<!--[-->");
              Sidebar_inset($$renderer4, {
                class: "h-svh overflow-hidden",
                children: ($$renderer5) => {
                  $$renderer5.push(`<header class="border-border flex h-11 shrink-0 items-center gap-2 border-b px-3">`);
                  if (Sidebar_trigger) {
                    $$renderer5.push("<!--[-->");
                    Sidebar_trigger($$renderer5, {});
                    $$renderer5.push("<!--]-->");
                  } else {
                    $$renderer5.push("<!--[!-->");
                    $$renderer5.push("<!--]-->");
                  }
                  $$renderer5.push(` <div class="bg-border h-4 w-px"></div> `);
                  if (current()) {
                    $$renderer5.push("<!--[0-->");
                    const Icon2 = current().icon;
                    if (Icon2) {
                      $$renderer5.push("<!--[-->");
                      Icon2($$renderer5, { class: "text-muted-foreground size-4" });
                      $$renderer5.push("<!--]-->");
                    } else {
                      $$renderer5.push("<!--[!-->");
                      $$renderer5.push("<!--]-->");
                    }
                    $$renderer5.push(` <span class="text-foreground text-sm font-medium">${escape_html(current().label)}</span> <span class="text-muted-foreground hidden truncate text-xs sm:inline">— ${escape_html(current().hint)}</span>`);
                  } else {
                    $$renderer5.push("<!--[-1-->");
                  }
                  $$renderer5.push(`<!--]--></header> <div class="min-h-0 flex-1 overflow-hidden">`);
                  if (descriptor.view) {
                    $$renderer5.push("<!--[0-->");
                    children($$renderer5);
                    $$renderer5.push(`<!---->`);
                  } else if (descriptor.error) {
                    $$renderer5.push("<!--[1-->");
                    $$renderer5.push(`<div class="grid h-full place-items-center p-6 text-center"><div class="max-w-md"><p class="text-foreground text-sm font-medium">Could not load the dataset descriptor</p> <p class="text-muted-foreground mt-1 text-xs">${escape_html(descriptor.error)}</p></div></div>`);
                  } else {
                    $$renderer5.push("<!--[-1-->");
                    $$renderer5.push(`<div class="text-muted-foreground grid h-full place-items-center text-sm">Loading dataset…</div>`);
                  }
                  $$renderer5.push(`<!--]--></div>`);
                },
                $$slots: { default: true }
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
export {
  _layout as default
};
