import { u as ssr_context } from "./renderer.js";
import { m as lifecycle_function_unavailable } from "./render-context.js";
import "clsx";
function onDestroy(fn) {
  /** @type {SSRContext} */
  ssr_context.r.on_destroy(fn);
}
function mount() {
  lifecycle_function_unavailable("mount");
}
function unmount() {
  lifecycle_function_unavailable("unmount");
}
async function tick() {
}
const SvelteSet = globalThis.Set;
const SvelteMap = globalThis.Map;
class MediaQuery {
  current;
  /**
   * @param {string} query
   * @param {boolean} [matches]
   */
  constructor(query, matches = false) {
    this.current = matches;
  }
}
function createSubscriber(_) {
  return () => {
  };
}
export {
  MediaQuery as M,
  SvelteSet as S,
  SvelteMap as a,
  createSubscriber as c,
  mount as m,
  onDestroy as o,
  tick as t,
  unmount as u
};
