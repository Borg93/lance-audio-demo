import "../../../chunks/async.js";
import "clsx";
import "../../../chunks/api.js";
import "@sveltejs/kit/internal";
import "../../../chunks/exports.js";
import "../../../chunks/utils.js";
import "@sveltejs/kit/internal/server";
import "../../../chunks/root.js";
import "../../../chunks/state.svelte.js";
import "../../../chunks/descriptor.js";
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    $$renderer2.push(`<div class="h-full w-full">`);
    {
      $$renderer2.push("<!--[-1-->");
      $$renderer2.push(`<div class="grid h-full place-items-center text-sm text-muted-foreground">Loading…</div>`);
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
export {
  _page as default
};
