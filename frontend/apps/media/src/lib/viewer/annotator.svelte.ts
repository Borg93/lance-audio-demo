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
import { type Table, tableFromIPC } from "apache-arrow";
import type { CommitShape, PixiContext, Tool } from "$lib/engine";
import { LayerStore } from "$lib/engine";
import type { LabelDelta, LabelOp, LabelOutcome, Selection } from "$lib/labeling/types";
import { PRODUCERS } from "$lib/labeling/producers";

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
  shape: string;
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

/** A newly drawn shape queued for the next Save (backend `NewAnnotation`). The chunk
 *  identity is stamped server-side, so we send only geometry + attributes. */
interface InsertRow {
  id: string;
  shape_type: string;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  polygon: number[];
  mask: string;
  label: string;
  text: string;
  group: string;
  status: string;
  source: string;
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
  saving = $state(false);
  saveError = $state<string | null>(null);
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

  // Structural edits queued for the next Save: shapes drawn (onCommit) and ids
  // deleted. Flushed by save() then reconciled by _reload() — no client-side Arrow
  // surgery (which would corrupt the index-keyed overlay). The sidebar reflects
  // deletes immediately (filtered from `rows`); new shapes render after save+reload.
  private _inserts = $state<InsertRow[]>([]);
  private _deletes = $state<string[]>([]);

  private _detachViewport: (() => void) | null = null;
  // POST target for Save (same URL the annotations are GET from). Null ⇒ read-only.
  private _saveUrl = $state<string | null>(null);

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
        shape: this._field(t, "shape_type", i) ?? "",
        group: this._field(t, "group", i) ?? "",
        text: this._field(t, "text", i) ?? "",
        source: this._field(t, "source", i) ?? "",
        confidence: this._num(t, "confidence", i),
        uncertainty: this._num(t, "uncertainty", i),
      });
    }
    // deletes reflect immediately in the sidebar (the canvas reconciles on save+reload)
    return this._deletes.length ? out.filter((r) => !this._deletes.includes(r.id)) : out;
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

  /** The review order — predictions first, highest uncertainty first (the active-
   *  learning queue). The ONE source both the sidebar list and accept-and-advance
   *  read, so "what to review next" is a sort, not a recompute. */
  readonly reviewQueue = $derived.by<AnnoRow[]>(() =>
    this.rows.toSorted((a, b) => {
      const ap = a.status === "prediction" ? 0 : 1;
      const bp = b.status === "prediction" ? 0 : 1;
      if (ap !== bp) return ap - bp;
      return (b.uncertainty ?? -1) - (a.uncertainty ?? -1);
    }),
  );

  /** 1-based position of the selection in the review queue (0 if none) + the total. */
  readonly queuePos = $derived.by<{ at: number; of: number }>(() => {
    const of = this.reviewQueue.length;
    const i = this.selectedIndex;
    const idx = i == null ? -1 : this.reviewQueue.findIndex((r) => r.index === i);
    return { at: idx < 0 ? 0 : idx + 1, of };
  });

  readonly canDraw = $derived(this.mode === "edit");
  readonly canUndo = $derived(this._undo.length > 0);
  readonly canRedo = $derived(this._redo.length > 0);
  /** Unsaved-edits flag: pending field edits, a canvas geometry edit, or queued
   *  structural inserts/deletes. */
  readonly dirty = $derived(
    this._undo.length > 0 || this._geoDirty || this._inserts.length > 0 || this._deletes.length > 0,
  );
  readonly canSave = $derived(this.dirty && !this.saving && this._saveUrl !== null);

  // ── engine lifecycle ──

  /** Called by a spatial viewer once its engine + data are ready. `saveUrl` (the
   *  annotations endpoint) enables the local-first Save; omit it for read-only. */
  attach(ctx: PixiContext, table: Table, saveUrl?: string): void {
    this.ctx = ctx;
    this.table = table;
    this.count = table.numRows;
    this._saveUrl = saveUrl ?? null;

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
    im.onCommit = (shape) => {
      this._inserts = [...this._inserts, this._buildInsert(shape)];
      this.count = ctx.plugins.arrow.getNumRows() + this._inserts.length;
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
    const i = this.selectedIndex;
    if (i == null) return;
    const t = this.table;
    const id = t ? this._raw(t, "id", i) : null;
    if (id) this._deletes = [...this._deletes, id]; // flushed on Save; sidebar drops it now
    this.ctx?.plugins.interaction.handleKeyDown("Delete");
    this._geoDirty = true;
    this.select(null);
  }

  /** Map a committed engine shape → the queued insert row (backend NewAnnotation). */
  private _buildInsert(shape: CommitShape): InsertRow {
    return {
      id: crypto.randomUUID(),
      shape_type: shape.type === "rect" ? "rectangle" : shape.type,
      x: shape.x,
      y: shape.y,
      width: shape.width,
      height: shape.height,
      rotation: shape.rotation ?? 0,
      polygon: shape.polygon ?? [],
      mask: shape.mask ?? "",
      label: "",
      text: "",
      group: "",
      status: "accepted",
      source: "human",
    };
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
    // Manual mode = ONE instance of the LabelOp abstraction (human · verdict ·
    // interactive · one). Routing through apply() proves the annotator isn't
    // coupled to the review flow — a model/batch producer slots into the same seam.
    this.apply({
      target: { level: "one", index },
      producer: "human",
      op: "verdict",
      execution: "interactive",
      payload: { fields: { status } },
    });
  }

  // ── review-queue navigation (accept-and-advance — the throughput loop) ──
  private _queuePos(): number {
    const i = this.selectedIndex;
    return i == null ? -1 : this.reviewQueue.findIndex((r) => r.index === i);
  }
  /** Move the selection by ±1 within the review queue (clamped). */
  selectQueueRelative(delta: number): void {
    const q = this.reviewQueue;
    if (!q.length) return;
    const pos = this._queuePos();
    const target = pos < 0 ? 0 : Math.min(Math.max(pos + delta, 0), q.length - 1);
    const row = q[target];
    if (row) this.select(row.index);
  }
  next(): void {
    this.selectQueueRelative(1);
  }
  prev(): void {
    this.selectQueueRelative(-1);
  }
  /** Set the selection's status, then advance to the next item still needing review
   *  (the next prediction in queue order), else the next row — the review loop. */
  acceptAndAdvance(status: string): void {
    const i = this.selectedIndex;
    if (i == null) return;
    const q = this.reviewQueue; // capture pre-change order (the row re-sorts after)
    const pos = q.findIndex((r) => r.index === i);
    this.setStatus(i, status);
    const rest = pos < 0 ? q : q.slice(pos + 1);
    const nextRow = rest.find((r) => r.status === "prediction") ?? rest[0];
    this.select(nextRow ? nextRow.index : null);
  }

  // ── the write-plane seam: dispatch a LabelOp (all 3 modes flow through here) ──
  apply(op: LabelOp): LabelOutcome {
    const spec = PRODUCERS[op.producer];
    if (!spec) return { status: "unsupported", reason: `unknown producer '${op.producer}'` };

    // Batch locus = a silver deriver over a (query/all) selection: the annotator
    // enqueues, the job surfaces async by media id + Lance version. Not wired in the
    // prototype (that's the lance-ray/catalog-mover write path).
    if (op.execution === "batch") {
      return {
        status: "queued",
        job: `${spec.source}:${op.op}`,
        note: "batch deriver (not wired)",
      };
    }

    // Interactive + human = the manual review path (real, local-first → Save).
    if (spec.kind === "human" && (op.op === "set" || op.op === "verdict")) {
      const fields = op.payload.fields ?? {};
      const deltas: LabelDelta[] = [];
      for (const index of this._resolveInteractive(op.target)) {
        for (const [field, value] of Object.entries(fields)) {
          this.updateField(index, field as EditableField, value);
        }
        deltas.push({ index, fields });
      }
      return { status: "applied", deltas };
    }

    // Interactive model/propagate (SAM click, INSID3-quick, DINO-on-a-page) — the
    // narrow interactive-assist exception; needs a predict/decode transport (follow-up).
    return {
      status: "unsupported",
      reason: `interactive ${spec.kind} '${spec.name}' not wired yet`,
    };
  }

  /** Resolve an interactive (annotation-level) Selection to engine row indices.
   *  Chunk-level selections are corpus-scale (batch), so client-side they fall back
   *  to the current canvas selection. */
  private _resolveInteractive(sel: Selection): number[] {
    switch (sel.level) {
      case "one":
        return [sel.index];
      case "picked":
        return sel.indices;
      default:
        return [...this.selectedSet];
    }
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

  // ── persistence: local-first Save → Lance merge_insert (NOT sync-per-edit) ──

  /** Flush the accumulated field-edit overlay to Lance as ONE atomic version, then
   *  reload so the display reflects the persisted (merge-reordered) rows. */
  async save(): Promise<void> {
    const t = this.table;
    const url = this._saveUrl;
    if (!t || !url || this.saving) return;

    const byIndex = new Map<number, Record<string, string>>();
    for (const [key, value] of this._overrides) {
      const sep = key.indexOf(":");
      const index = Number(key.slice(0, sep));
      const row = byIndex.get(index) ?? {};
      row[key.slice(sep + 1)] = value;
      byIndex.set(index, row);
    }
    const edits = [...byIndex].map(([index, fields]) => ({
      id: this._raw(t, "id", index) ?? String(index),
      ...fields,
    }));
    if (edits.length === 0 && this._inserts.length === 0 && this._deletes.length === 0) return;

    this.saving = true;
    this.saveError = null;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ edits, inserts: this._inserts, deletes: this._deletes }),
      });
      if (!res.ok) throw new Error(`save failed (HTTP ${res.status})`);
      await this._reload();
      this._undo = [];
      this._redo = [];
      this._inserts = [];
      this._deletes = [];
      this._geoDirty = false;
    } catch (e) {
      this.saveError = e instanceof Error ? e.message : String(e);
    } finally {
      this.saving = false;
    }
  }

  /** Re-fetch the persisted annotations into the canvas + controller, clearing the
   *  now-flushed overlay. merge_insert reorders rows, so selection is dropped. */
  private async _reload(): Promise<void> {
    const url = this._saveUrl;
    const ctx = this.ctx;
    if (!url || !ctx) return;
    const res = await fetch(url);
    if (!res.ok) return;
    const table = tableFromIPC(new Uint8Array(await res.arrayBuffer()));
    this._overrides.clear();
    ctx.plugins.arrow.load(table);
    ctx.plugins.arrow.sync();
    this.table = table;
    this.count = table.numRows;
    this.select(null);
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
