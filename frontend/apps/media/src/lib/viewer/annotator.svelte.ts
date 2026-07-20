/**
 * AnnotatorController — the reactive facade the ra-anno layout binds to.
 *
 * The engine core (InteractionManager / ArrowDataPlugin / ImagePlugin) is
 * framework-agnostic and lives locked inside PixiCanvas's context, reachable
 * only from the `onready` ctx. ra-anno's original layout assumed a Svelte
 * binding layer (`$lib/stores/*.svelte.ts`) that was never ported. This class
 * IS that layer: a route-level, Svelte-5-runes source of truth that
 *   1. captures the engine handle (via `attach`),
 *   2. mirrors the engine's plain getters + callbacks into reactive state so
 *      dumb/controlled layout components (Toolbar, Sidebar, ZoomControls, …)
 *      stay reactive, and
 *   3. owns the layer grouping + the annotation table the sidebar reads.
 *
 * Modality-neutral: spatial (image/video-frame) viewers `attach` a PixiContext;
 * a temporal (audio) viewer can attach its own control surface to the same
 * controller later. The layout binds to the controller, never to the engine.
 */
import { SvelteMap, SvelteSet } from "svelte/reactivity";
import type { Table } from "apache-arrow";
import type { PixiContext, Tool } from "$lib/engine";
import { LayerStore } from "$lib/engine";

export type Mode = "view" | "edit";

export interface BrushOptions {
  radius: number;
  erasing: boolean;
  maskMode: "instance" | "semantic";
  output: "mask" | "polygon";
}

/** One annotation row projected into the flat shape the sidebar/list render.
 *  Identity is the engine's ROW INDEX (the engine is index-based); `id` is just
 *  a display field. Overlay edits win over the server table. */
export interface AnnoRow {
  index: number;
  id: string;
  label: string;
  status: string;
  group: string;
  text: string;
  source: string;
  confidence: number | null;
  uncertainty: number | null;
}

/** Fields the sidebar can edit inline. */
export type EditableField = "label" | "status" | "group" | "text";

/** One reversible field edit (relabel / status / text / group) — the unit of
 *  undo/redo. `before`/`after` are the effective (overlay-aware) string values. */
interface FieldEdit {
  index: number;
  field: EditableField;
  before: string;
  after: string;
}

const STRING_FIELD_CANDIDATES = ["label", "status", "source", "group", "reviewer"];

/** #rrggbb ⇄ 0xRRGGBB — self-contained so the panel needs no engine color util. */
export function numToHex(n: number): string {
  return "#" + (n & 0xffffff).toString(16).padStart(6, "0");
}
export function hexToNum(hex: string): number {
  return parseInt(hex.replace(/^#/, ""), 16) & 0xffffff;
}

export class AnnotatorController {
  // ── engine handle + data ──
  ctx = $state<PixiContext | null>(null);
  table = $state<Table | null>(null);

  // ── mirrored engine state ──
  mode = $state<Mode>("edit");
  activeTool = $state<Tool>("select");
  selectedIndex = $state<number | null>(null);
  readonly selectedSet = new SvelteSet<number>();
  zoomPercent = $state(1);
  count = $state(0);
  brushOptions = $state<BrushOptions>({
    radius: 20,
    erasing: false,
    maskMode: "instance",
    output: "mask",
  });

  // ── layer grouping (mirrored from LayerStore) ──
  readonly layers = new LayerStore();
  groupByColumn = $state("label");
  readonly hiddenGroups = new SvelteSet<string>();
  readonly groupColors = new SvelteMap<string, number>();

  // Local field overlay for sidebar/list display, keyed `${index}:${field}` (the
  // canvas is updated separately via arrow.setFieldOverride). SvelteMap ⇒ edits
  // re-derive `rows` with no manual version counter.
  private readonly _overrides = new SvelteMap<string, string>();

  // Undo/redo of field edits (the review operations: relabel / accept / reject /
  // text). Geometry dirtiness is tracked separately (the engine owns geometry).
  private _undo = $state<FieldEdit[]>([]);
  private _redo = $state<FieldEdit[]>([]);
  private _geoDirty = $state(false);

  private _detachViewport: (() => void) | null = null;

  constructor() {
    this.layers.on(() => this._pullLayers());
    this._pullLayers();
  }

  // ── derived views the layout reads ──

  /** String columns available to group by. */
  readonly groupColumns = $derived.by<string[]>(() => {
    const t = this.table;
    if (!t) return [];
    const names = new Set(t.schema.fields.map((f) => f.name));
    return STRING_FIELD_CANDIDATES.filter((c) => names.has(c));
  });

  /** Flat rows for the sidebar list, overlay-aware. */
  readonly rows = $derived.by<AnnoRow[]>(() => {
    const t = this.table;
    if (!t) return [];
    const out: AnnoRow[] = [];
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
        uncertainty: this._num(t, "uncertainty", i),
      });
    }
    return out;
  });

  /** Distinct groups for the current group-by column, with counts. */
  readonly groups = $derived.by<{ name: string; count: number }[]>(() => {
    const col = this.groupByColumn;
    const counts = new Map<string, number>();
    for (const r of this.rows) {
      const key = (r as unknown as Record<string, string>)[col] ?? "";
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts.entries()]
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => a.name.localeCompare(b.name));
  });

  /** The currently selected row (single selection), overlay-aware. */
  readonly selected = $derived.by<AnnoRow | null>(() => {
    const i = this.selectedIndex;
    if (i == null) return null;
    return this.rows.find((r) => r.index === i) ?? null;
  });

  readonly canDraw = $derived(this.mode === "edit");
  readonly canUndo = $derived(this._undo.length > 0);
  readonly canRedo = $derived(this._redo.length > 0);
  /** Unsaved-edits flag: any pending field edit OR a geometry edit on the canvas. */
  readonly dirty = $derived(this._undo.length > 0 || this._geoDirty);

  // ── engine lifecycle ──

  /** Called by a spatial viewer once its engine + data are ready. */
  attach(ctx: PixiContext, table: Table): void {
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
      if (hasDirty) this._geoDirty = true;
    };
    im.onCommit = () => {
      this.count = ctx.plugins.arrow.getNumRows();
      this._geoDirty = true;
    };

    // chain (never overwrite) the image viewport hook PixiCanvas installed, so
    // zoomPercent tracks wheel-zoom + pan + our zoom buttons alike.
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

  detach(): void {
    this._detachViewport?.();
    this._detachViewport = null;
    this.ctx = null;
    this.table = null;
  }

  // ── toolbar / mode ──
  setTool(tool: Tool): void {
    this.activeTool = tool;
    this.ctx?.plugins.interaction.setTool(tool);
  }
  toggleMode(): void {
    this.mode = this.mode === "edit" ? "view" : "edit";
    this.ctx?.plugins.interaction.setEditMode(this.mode === "edit");
    if (this.mode === "view") this.setTool("select");
  }
  setBrushOptions(patch: Partial<BrushOptions>): void {
    this.brushOptions = { ...this.brushOptions, ...patch };
    this.ctx?.plugins.interaction.setBrushOptions(this.brushOptions);
  }

  // ── selection ──
  select(index: number | null): void {
    this.selectedIndex = index;
    const im = this.ctx?.plugins.interaction;
    im?.select(index);
    this._mirrorSelection(im?.getSelectedSet() ?? new Set());
  }
  deleteSelected(): void {
    if (this.selectedIndex == null) return;
    this.ctx?.plugins.interaction.handleKeyDown("Delete");
    this._geoDirty = true;
    this.select(null);
  }
  convertToPolygon(): void {
    if (this.ctx?.plugins.interaction.convertToPolygon()) this._geoDirty = true;
  }

  // ── inline field edits (canvas + overlay) + undo/redo ──
  updateField(index: number, field: EditableField, value: string): void {
    const t = this.table;
    if (!t) return;
    const before = this._field(t, field, index) ?? "";
    if (before === value) return;
    this._undo = [...this._undo, { index, field, before, after: value }];
    this._redo = [];
    this._setField(index, field, value);
  }
  setStatus(index: number, status: string): void {
    this.updateField(index, "status", status);
  }

  /** Revert the last field edit (canvas + overlay), then re-select it. */
  undo(): void {
    const op = this._undo.at(-1);
    if (!op) return;
    this._undo = this._undo.slice(0, -1);
    this._redo = [...this._redo, op];
    this._setField(op.index, op.field, op.before);
    this.select(op.index);
  }
  /** Re-apply the last undone field edit. */
  redo(): void {
    const op = this._redo.at(-1);
    if (!op) return;
    this._redo = this._redo.slice(0, -1);
    this._undo = [...this._undo, op];
    this._setField(op.index, op.field, op.after);
    this.select(op.index);
  }

  /** Apply a field value to BOTH the WebGPU canvas (arrow.setFieldOverride →
   *  re-render) and the sidebar overlay. The single write path for edit/undo/redo. */
  private _setField(index: number, field: EditableField, value: string): void {
    this._overrides.set(`${index}:${field}`, value);
    this.ctx?.plugins.arrow.setFieldOverride(index, field, value);
    this.ctx?.plugins.arrow.sync();
  }

  // ── zoom ──
  zoomIn(): void {
    this.ctx?.plugins.image.zoomIn();
  }
  zoomOut(): void {
    this.ctx?.plugins.image.zoomOut();
  }
  resetView(): void {
    this.ctx?.plugins.image.resetView();
  }

  // ── layers ──
  setGroupBy(column: string): void {
    this.layers.setGroupBy(column);
    this.groupByColumn = column;
  }
  toggleGroupVisible(group: string): void {
    this.layers.toggleVisibility(group);
    this.ctx?.plugins.arrow.setGroupVisible(group, !this.layers.isHidden(group));
  }
  setGroupColor(group: string, hex: string): void {
    this.layers.setColor(group, hexToNum(hex));
  }
  isHidden(group: string): boolean {
    return this.hiddenGroups.has(group);
  }
  groupColorHex(group: string): string {
    const n = this.groupColors.get(group);
    return n == null ? "#3b82f6" : numToHex(n);
  }

  // ── internals ──
  private _mirrorSelection(set: ReadonlySet<number>): void {
    this.selectedSet.clear();
    for (const i of set) this.selectedSet.add(i);
  }
  private _pullLayers(): void {
    this.groupByColumn = this.layers.groupByColumn;
    this.hiddenGroups.clear();
    for (const g of this.layers.hiddenGroups) this.hiddenGroups.add(g);
    this.groupColors.clear();
    for (const [k, v] of this.layers.groupColors) this.groupColors.set(k, v);
    this._syncLayerConfig();
  }
  private _syncLayerConfig(): void {
    const arrow = this.ctx?.plugins.arrow;
    if (!arrow) return;
    arrow.setLayerConfig({
      hiddenGroups: this.layers.hiddenGroups,
      groupByColumn: this.layers.groupByColumn,
      groupColors: this.layers.groupColors,
    });
    arrow.sync();
  }

  private _raw(t: Table, field: string, i: number): string | null {
    const v = t.getChild(field)?.get(i);
    return v == null ? null : String(v);
  }
  private _field(t: Table, field: string, i: number): string | null {
    const o = this._overrides.get(`${i}:${field}`);
    if (o != null) return o;
    return this._raw(t, field, i);
  }
  private _num(t: Table, field: string, i: number): number | null {
    const v = t.getChild(field)?.get(i);
    return typeof v === "number" ? v : null;
  }
}
