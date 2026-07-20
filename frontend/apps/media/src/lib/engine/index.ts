/**
 * ra-anno engine — public API.
 *
 * Framework-agnostic annotation + rendering engine: PixiJS (WebGPU/WebGL) rendering,
 * Apache Arrow data layer, interaction/tools/editors, geometry, mask ops, and the
 * annotation schema. Imports ZERO Svelte / `$app` — consumed by the thin Svelte
 * binding (`$lib/svelte`) and the app. Intended to graduate into a standalone package.
 */

// ── Rendering (PixiJS plugins + context types) ──
export * from "./pixi/ImagePlugin.js";
export * from "./pixi/ArrowDataPlugin.js";
export * from "./pixi/types.js";

// ── Interaction layer ──
export * from "./interaction/InteractionManager.js";
export * from "./interaction/geometry.js";
export * from "./interaction/types.js";

// ── Drawing tools ──
export * from "./tools/RectTool.js";
export * from "./tools/PolygonTool.js";
export * from "./tools/LassoTool.js";
export * from "./tools/MagneticTool.js";
export * from "./tools/ScissorsTool.js";
export * from "./tools/PencilTool.js";
export * from "./tools/PointTool.js";
export * from "./tools/LineTool.js";
export * from "./tools/BrushTool.js";

// ── Shape editors ──
export * from "./editors/RectEditor.js";
export * from "./editors/PolygonEditor.js";

// ── Framework-agnostic state stores (plain TS + observer; Svelte binding adapts them) ──
export * from "./store/AnnotationStore.js";
export * from "./store/LayerStore.js";

// ── Data model + utilities ──
export * from "./schema.js";
export * from "./utils/arrow.js";
export * from "./utils/color.js";
export * from "./maskOps.js";

// The public `Tool` is the tool-id union from ./pixi/types; the engine-internal `Tool`
// *interface* in ./interaction/types is imported directly by tool classes, not re-exported.
export type { Tool } from "./pixi/types.js";
