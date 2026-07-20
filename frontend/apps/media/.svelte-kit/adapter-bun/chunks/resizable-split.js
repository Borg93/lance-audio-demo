import "./async.js";
import { s as sanitize_props, a as spread_props, b as slot, i as attr_class, n as attr_style, d as derived } from "./renderer.js";
import { I as Icon } from "./Icon.js";
import { r as run } from "./render-context.js";
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
  Resizable_split as R,
  X
};
