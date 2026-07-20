import {
  type Data,
  DataType,
  makeBuilder,
  makeData,
  RecordBatch,
  Schema,
  Struct,
  type Table,
  Table as ArrowTable,
  tableFromArrays,
  tableFromIPC,
  tableToIPC,
} from "apache-arrow";
import type { AnnotationTransport } from "./transport.js";
// NOTE: makeTable (zero-copy for typed arrays) can't be used here because our
// tables have mixed types (Float32 + Utf8 + List). tableFromArrays handles
// type inference but always copies. The overlay pattern avoids rebuilds for
// field edits — tableFromArrays only runs on rematerialize (sidebar) and save.
//
// Framework-agnostic: NO Svelte runes. Reactivity is delivered to the Svelte
// binding via the observer (`on`/`emit`); see src/lib/stores/annotations.svelte.ts.

/** Change signal kinds. "structural" = append/delete/load/save/undo (Pixi re-syncs);
 *  "field" = field-only edits (sidebar reacts, Pixi does NOT re-sync). */
export type StoreChangeKind = "structural" | "field";

// ── Helpers ──

/** Check if a field is a List type (e.g. polygon: List<Float32>) */
function isListField(field: { type: DataType }): boolean {
  return DataType.isList(field.type);
}

/** Convert Arrow List element to plain JS array */
function arrowListToArray(val: unknown): number[] | null {
  if (!val || typeof val !== "object") return null;
  const v = val as { length: number; get: (i: number) => number };
  if (!v.length || v.length === 0) return null;
  const arr = new Array(v.length);
  for (let j = 0; j < v.length; j++) arr[j] = v.get(j);
  return arr;
}

/** Read a column value, converting List types to plain arrays */
function readColumnValue(col: import("apache-arrow").Vector, i: number, isList: boolean): unknown {
  if (isList) return arrowListToArray(col.get(i));
  return col.get(i);
}

/** Build full table from Arrow base + overlays */
function materializeTable(
  base: Table,
  fieldOverrides: Map<number, Map<string, unknown>>,
  appendedRows: Record<string, unknown>[],
  deletedIndices: Set<number>,
): Table {
  const n = base.numRows + appendedRows.length - deletedIndices.size;
  const cols: Record<string, unknown[]> = {};
  // Pre-check which fields are List type (avoid per-row string comparison)
  const listFlags = base.schema.fields.map((f) => isListField(f));

  for (const field of base.schema.fields) {
    cols[field.name] = new Array(n);
  }

  let dst = 0;
  for (let i = 0; i < base.numRows; i++) {
    if (deletedIndices.has(i)) continue;
    const rowOverrides = fieldOverrides.get(i);
    for (let f = 0; f < base.schema.fields.length; f++) {
      const field = base.schema.fields[f];
      const override = rowOverrides?.get(field.name);
      if (override !== undefined) {
        cols[field.name][dst] = override;
      } else {
        cols[field.name][dst] = readColumnValue(base.getChild(field.name)!, i, listFlags[f]);
      }
    }
    dst++;
  }
  for (const row of appendedRows) {
    for (const field of base.schema.fields) {
      cols[field.name][dst] = row[field.name] ?? null;
    }
    dst++;
  }

  const raw = tableFromArrays(cols);
  if (base.schema.metadata.size > 0) {
    return new ArrowTable(new Schema(raw.schema.fields, base.schema.metadata), raw.batches);
  }
  return raw;
}

/**
 * Build a Table containing only `rows`, using `schema`'s EXACT field types
 * (Float32 stays Float32, Dictionary<Utf8> appends a delta, List<Float32> kept).
 * Because the schema matches the base, the result `concat`s onto it cleanly —
 * O(rows), not O(total). Used by interactive appends AND streamed inference.
 */
function buildBatchTable(schema: Schema, rows: Record<string, unknown>[]): Table {
  const children: Data[] = [];
  for (const field of schema.fields) {
    const builder = makeBuilder({
      type: field.type,
      nullValues: [null, undefined],
    });
    for (const row of rows) builder.append(row[field.name] ?? null);
    children.push(builder.finish().flush());
  }
  const structData = makeData({
    type: new Struct(schema.fields),
    length: rows.length,
    children,
  });
  return new ArrowTable(schema, new RecordBatch(schema, structData));
}

/** Empty annotation table carrying the FULL ANNOTATION_COLUMNS schema (see schema.ts).
 *  Every column must be present or materializeTable()/append drops it. */
function makeEmptyTable(): Table {
  return tableFromArrays({
    id: [] as string[],
    page_id: [] as string[],
    dataset_id: [] as string[],
    shape_type: [] as string[],
    x: new Float32Array(0),
    y: new Float32Array(0),
    width: new Float32Array(0),
    height: new Float32Array(0),
    rotation: new Float32Array(0),
    polygon: [] as (number[] | null)[],
    text: [] as string[],
    label: [] as string[],
    confidence: new Float32Array(0),
    source: [] as string[],
    status: [] as string[],
    reviewer: [] as string[],
    group: [] as string[],
    group_id: [] as string[],
    reading_order: new Int32Array(0),
    difficult: [] as boolean[],
    links: [] as string[],
    mask: [] as string[],
    metadata: [] as string[],
  });
}

/** Extract specific rows from a materialized table for delta save */
function extractRows(table: Table, indices: number[]): Table {
  if (indices.length === 0) return tableFromArrays({});
  const listFlags = table.schema.fields.map((f) => isListField(f));
  const cols: Record<string, unknown[]> = {};
  for (let f = 0; f < table.schema.fields.length; f++) {
    const field = table.schema.fields[f];
    const col = table.getChild(field.name)!;
    const arr: unknown[] = new Array(indices.length);
    for (let j = 0; j < indices.length; j++) {
      arr[j] = readColumnValue(col, indices[j], listFlags[f]);
    }
    cols[field.name] = arr;
  }
  return tableFromArrays(cols);
}

// ── Undo snapshot ──

interface UndoSnapshot {
  fieldOverrides: Map<number, Map<string, unknown>>;
  appendedRows: Record<string, unknown>[];
  deletedIndices: Set<number>;
  deletedIds: Set<string>;
}

function cloneSnapshot(
  fo: Map<number, Map<string, unknown>>,
  ar: Record<string, unknown>[],
  di: Set<number>,
  dids: Set<string>,
): UndoSnapshot {
  return {
    fieldOverrides: new Map([...fo].map(([k, v]) => [k, new Map(v)])),
    appendedRows: ar.map((r) => ({ ...r })),
    deletedIndices: new Set(di),
    deletedIds: new Set(dids),
  };
}

// ── Store ──

export class AnnotationStore {
  private _serverTables: Record<string, Table> = {};
  private _fieldOverrides: Record<string, Map<number, Map<string, unknown>>> = {};
  private _appendedRows: Record<string, Record<string, unknown>[]> = {};
  private _deletedIndices: Record<string, Set<number>> = {};
  private _deletedIds: Record<string, Set<string>> = {};
  private _undoStack: Record<string, UndoSnapshot[]> = {};
  private _redoStack: Record<string, UndoSnapshot[]> = {};
  private _versions: Record<string, string> = {};
  private _dirty: Record<string, boolean> = {};
  private _loading: Record<string, boolean> = {};
  private _materializedTables: Record<string, Table> = {};
  private _structuralVersion: Record<string, number> = {};
  // Bumps on every field edit — sidebar watches this for reactivity
  private _fieldVersion: Record<string, number> = {};

  // ── Observer (framework-agnostic reactivity bridge) ──
  private _listeners = new Set<(kind: StoreChangeKind) => void>();

  /**
   * @param transport persistence backend (HTTP, in-memory, gRPC, an inference
   * service that streams predictions, …). Keeps the engine framework-/host-
   * agnostic — the Svelte binding injects an HTTP transport.
   */
  constructor(private readonly transport: AnnotationTransport) {}

  /** Subscribe to change signals. Returns an unsubscribe fn. */
  on(listener: (kind: StoreChangeKind) => void): () => void {
    this._listeners.add(listener);
    return () => this._listeners.delete(listener);
  }

  private emit(kind: StoreChangeKind): void {
    for (const listener of this._listeners) listener(kind);
  }

  /** Materialized table — only accurate after structural changes.
   *  For field-only reads, use getFieldValue() which checks overlays. */
  table(pageId: string): Table | null {
    return this._materializedTables[pageId] ?? null;
  }

  /** Field version — increments on every field edit. Watch this for sidebar reactivity. */
  fieldVersion(pageId: string): number {
    return this._fieldVersion[pageId] ?? 0;
  }

  /** Read a field value — checks overlay first, falls back to server table.
   *  O(1) per call, no table rebuild needed. */
  getFieldValue(pageId: string, rowIndex: number, field: string): unknown {
    // Check field overlay — O(1)
    const override = this._fieldOverrides[pageId]?.get(rowIndex)?.get(field);
    if (override !== undefined) return override;

    const base = this._serverTables[pageId];
    if (!base) return null;
    const deleted = this._deletedIndices[pageId];
    const appended = this._appendedRows[pageId] ?? [];

    // Fast path: no deletions — materialized index maps directly to base/appended
    if (!deleted || deleted.size === 0) {
      if (rowIndex < base.numRows) {
        return base.getChild(field)?.get(rowIndex);
      }
      const appendedIdx = rowIndex - base.numRows;
      if (appendedIdx >= 0 && appendedIdx < appended.length) {
        return appended[appendedIdx][field] ?? null;
      }
      return null;
    }

    // Slow path: has deletions — walk base to find materialized index
    let matIdx = 0;
    for (let i = 0; i < base.numRows; i++) {
      if (deleted.has(i)) continue;
      if (matIdx === rowIndex) {
        return base.getChild(field)?.get(i);
      }
      matIdx++;
    }
    const appendedIdx = rowIndex - matIdx;
    if (appendedIdx >= 0 && appendedIdx < appended.length) {
      return appended[appendedIdx][field] ?? null;
    }
    return null;
  }

  serverTable(pageId: string): Table | null {
    return this._serverTables[pageId] ?? null;
  }

  structuralVersion(pageId: string): number {
    return this._structuralVersion[pageId] ?? 0;
  }

  isDirty(pageId: string): boolean {
    return this._dirty[pageId] ?? false;
  }

  isLoading(pageId: string): boolean {
    return this._loading[pageId] ?? false;
  }

  version(pageId: string): string | null {
    return this._versions[pageId] ?? null;
  }

  get canUndo(): boolean {
    return Object.values(this._undoStack).some((s) => s.length > 0);
  }

  get canRedo(): boolean {
    return Object.values(this._redoStack).some((s) => s.length > 0);
  }

  // ── Private helpers ──

  private getOverlays(pageId: string) {
    return {
      fields: this._fieldOverrides[pageId] ?? new Map(),
      appended: this._appendedRows[pageId] ?? [],
      deleted: this._deletedIndices[pageId] ?? new Set(),
      deletedIds: this._deletedIds[pageId] ?? new Set(),
    };
  }

  private pushUndo(pageId: string): void {
    const { fields, appended, deleted, deletedIds } = this.getOverlays(pageId);
    const snapshot = cloneSnapshot(fields, appended, deleted, deletedIds);
    const stack = this._undoStack[pageId] ?? [];
    stack.push(snapshot);
    this._undoStack = { ...this._undoStack, [pageId]: stack };
    this._redoStack = { ...this._redoStack, [pageId]: [] };
  }

  private bumpStructural(pageId: string): void {
    const v = this._structuralVersion[pageId] ?? 0;
    this._structuralVersion = {
      ...this._structuralVersion,
      [pageId]: v + 1,
    };
    this.emit("structural");
  }

  private bumpFieldVersion(pageId: string): void {
    const v = this._fieldVersion[pageId] ?? 0;
    this._fieldVersion = { ...this._fieldVersion, [pageId]: v + 1 };
    this.emit("field");
  }

  private rematerialize(pageId: string): void {
    const base = this._serverTables[pageId];
    if (!base) return;
    const { fields, appended, deleted } = this.getOverlays(pageId);
    const table = materializeTable(base, fields, appended, deleted);
    this._materializedTables = {
      ...this._materializedTables,
      [pageId]: table,
    };
    this._dirty = { ...this._dirty, [pageId]: true };
  }

  private applyUndoSnapshot(pageId: string, snapshot: UndoSnapshot): void {
    this._fieldOverrides = {
      ...this._fieldOverrides,
      [pageId]: snapshot.fieldOverrides,
    };
    this._appendedRows = {
      ...this._appendedRows,
      [pageId]: snapshot.appendedRows,
    };
    this._deletedIndices = {
      ...this._deletedIndices,
      [pageId]: snapshot.deletedIndices,
    };
    this._deletedIds = {
      ...this._deletedIds,
      [pageId]: snapshot.deletedIds,
    };
    this.bumpStructural(pageId);
    this.rematerialize(pageId);
  }

  private clearOverlays(pageId: string): void {
    this._fieldOverrides = {
      ...this._fieldOverrides,
      [pageId]: new Map(),
    };
    this._appendedRows = { ...this._appendedRows, [pageId]: [] };
    this._deletedIndices = {
      ...this._deletedIndices,
      [pageId]: new Set(),
    };
    this._deletedIds = { ...this._deletedIds, [pageId]: new Set() };
    this._undoStack = { ...this._undoStack, [pageId]: [] };
    this._redoStack = { ...this._redoStack, [pageId]: [] };
    this._dirty = { ...this._dirty, [pageId]: false };
    this.bumpStructural(pageId);
  }

  // ── Load ──

  async load(pageId: string): Promise<Table> {
    const cached = this._serverTables[pageId];
    if (cached) return this._materializedTables[pageId] ?? cached;

    this._loading = { ...this._loading, [pageId]: true };
    this.emit("structural");

    try {
      const result = await this.transport.load(pageId);

      // No annotations yet — empty table carrying the full schema.
      if (!result) {
        const empty = makeEmptyTable();
        this._serverTables = { ...this._serverTables, [pageId]: empty };
        this._materializedTables = {
          ...this._materializedTables,
          [pageId]: empty,
        };
        this.emit("structural");
        return empty;
      }

      this._versions = { ...this._versions, [pageId]: result.etag };

      // Whole-buffer path
      if (result.bytes) {
        const table = tableFromIPC(result.bytes);
        this._serverTables = { ...this._serverTables, [pageId]: table };
        this._materializedTables = {
          ...this._materializedTables,
          [pageId]: table,
        };
        this.clearOverlays(pageId);
        return table;
      }

      // Streaming path — parse RecordBatches progressively, render as they arrive.
      // This is the same path large model-inference results stream through.
      if (result.stream) {
        const reader = result.stream.getReader();
        const chunks: Uint8Array[] = [];
        let totalLen = 0;
        let table: Table | null = null;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          totalLen += value.length;
          try {
            const partial = tableFromIPC(concatBytes(chunks, totalLen));
            if (partial.numRows > 0) {
              table = partial;
              this._serverTables = { ...this._serverTables, [pageId]: partial };
              this._materializedTables = {
                ...this._materializedTables,
                [pageId]: partial,
              };
              this.emit("structural");
            }
          } catch {
            /* incomplete chunk — wait for more */
          }
        }
        if (chunks.length > 0) {
          table = tableFromIPC(concatBytes(chunks, totalLen));
          this._serverTables = { ...this._serverTables, [pageId]: table };
          this._materializedTables = {
            ...this._materializedTables,
            [pageId]: table,
          };
        }
        this.clearOverlays(pageId);
        return table ?? makeEmptyTable();
      }

      // Transport returned neither bytes nor a stream — treat as empty.
      const empty = makeEmptyTable();
      this._serverTables = { ...this._serverTables, [pageId]: empty };
      this._materializedTables = {
        ...this._materializedTables,
        [pageId]: empty,
      };
      return empty;
    } finally {
      this._loading = { ...this._loading, [pageId]: false };
    }
  }

  // ── Edits ──

  /** Field edit — O(1) overlay, rematerializes for sidebar but does NOT bump structural */
  updateLocal(pageId: string, rowIndex: number, updates: Record<string, unknown>): void {
    this.pushUndo(pageId);
    const fields = this._fieldOverrides[pageId] ?? new Map();
    let row = fields.get(rowIndex);
    if (!row) {
      row = new Map();
      fields.set(rowIndex, row);
    }
    for (const [k, v] of Object.entries(updates)) {
      row.set(k, v);
    }
    this._fieldOverrides = { ...this._fieldOverrides, [pageId]: fields };
    this._dirty = { ...this._dirty, [pageId]: true };
    this.bumpFieldVersion(pageId);
    // NOTE: no bumpStructural — Pixi uses setFieldOverride for rendering changes
    // NOTE: no rematerialize — sidebar reads via getFieldValue() for O(1) field access
  }

  /** Batch field edit — same as updateLocal but for multiple rows */
  batchUpdateLocal(pageId: string, updatesMap: Map<number, Record<string, unknown>>): void {
    if (updatesMap.size === 0) return;
    this.pushUndo(pageId);
    const fields = this._fieldOverrides[pageId] ?? new Map();
    for (const [rowIndex, updates] of updatesMap) {
      let row = fields.get(rowIndex);
      if (!row) {
        row = new Map();
        fields.set(rowIndex, row);
      }
      for (const [k, v] of Object.entries(updates)) {
        row.set(k, v);
      }
    }
    this._fieldOverrides = { ...this._fieldOverrides, [pageId]: fields };
    this._dirty = { ...this._dirty, [pageId]: true };
    this.bumpFieldVersion(pageId);
    // NOTE: no bumpStructural
    // NOTE: no rematerialize — sidebar reads via getFieldValue() for O(1) field access
  }

  /** Append one row — structural change, bumps version. */
  appendLocal(pageId: string, row: Record<string, unknown>): Table | null {
    return this.appendManyLocal(pageId, [row]);
  }

  /**
   * Append many rows in ONE RecordBatch — O(appended), NOT O(total). The old
   * path rebuilt the whole table on every shape (≈1.3s to add 1 row to 100k);
   * this concats a single new batch onto the existing one instead. Used by
   * drawing commits and by streamed model inference (boxes/points/masks → rows).
   */
  appendManyLocal(pageId: string, rows: Record<string, unknown>[]): Table | null {
    const base = this._serverTables[pageId];
    if (!base) return null;
    if (rows.length === 0) return this._materializedTables[pageId] ?? null;
    this.pushUndo(pageId);
    const appended = [...(this._appendedRows[pageId] ?? []), ...rows];
    this._appendedRows = { ...this._appendedRows, [pageId]: appended };
    this.bumpStructural(pageId);

    // Fast path: concat the new batch onto the current materialized table.
    // Fall back to a full rematerialize only when field overlays exist, so the
    // next Pixi re-sync keeps edited colors baked in (rare on the bulk path).
    const fields = this._fieldOverrides[pageId];
    const current = this._materializedTables[pageId];
    if (current && (!fields || fields.size === 0)) {
      const batch = buildBatchTable(current.schema, rows);
      this._materializedTables = {
        ...this._materializedTables,
        [pageId]: current.concat(batch),
      };
      this._dirty = { ...this._dirty, [pageId]: true };
    } else {
      this.rematerialize(pageId);
    }
    return this._materializedTables[pageId] ?? null;
  }

  /** Delete row — structural change, bumps version */
  deleteLocal(pageId: string, rowIndex: number): void {
    const base = this._serverTables[pageId];
    if (!base) return;
    this.pushUndo(pageId);

    const baseRows = base.numRows;
    if (rowIndex < baseRows) {
      const deleted = new Set(this._deletedIndices[pageId] ?? []);
      deleted.add(rowIndex);
      this._deletedIndices = { ...this._deletedIndices, [pageId]: deleted };

      const id = String(base.getChild("id")?.get(rowIndex) ?? "");
      if (id) {
        const deletedIds = new Set(this._deletedIds[pageId] ?? []);
        deletedIds.add(id);
        this._deletedIds = { ...this._deletedIds, [pageId]: deletedIds };
      }

      const fields = this._fieldOverrides[pageId];
      if (fields) {
        fields.delete(rowIndex);
        this._fieldOverrides = { ...this._fieldOverrides, [pageId]: fields };
      }
    } else {
      const appended = [...(this._appendedRows[pageId] ?? [])];
      const deleted = this._deletedIndices[pageId] ?? new Set();
      const actualAppendedIdx = rowIndex - baseRows + deleted.size;
      if (actualAppendedIdx >= 0 && actualAppendedIdx < appended.length) {
        appended.splice(actualAppendedIdx, 1);
        this._appendedRows = { ...this._appendedRows, [pageId]: appended };
      }
    }

    this.bumpStructural(pageId);
    this.rematerialize(pageId);
  }

  // ── Undo / Redo ──

  undo(pageId: string): Table | null {
    const stack = this._undoStack[pageId];
    if (!stack || stack.length === 0) return null;

    const { fields, appended, deleted, deletedIds } = this.getOverlays(pageId);
    const current = cloneSnapshot(fields, appended, deleted, deletedIds);
    const redo = this._redoStack[pageId] ?? [];
    redo.push(current);
    this._redoStack = { ...this._redoStack, [pageId]: redo };

    const prev = stack.pop()!;
    this._undoStack = { ...this._undoStack, [pageId]: stack };
    this.applyUndoSnapshot(pageId, prev);
    return this._materializedTables[pageId] ?? null;
  }

  redo(pageId: string): Table | null {
    const stack = this._redoStack[pageId];
    if (!stack || stack.length === 0) return null;

    const { fields, appended, deleted, deletedIds } = this.getOverlays(pageId);
    const current = cloneSnapshot(fields, appended, deleted, deletedIds);
    const undo = this._undoStack[pageId] ?? [];
    undo.push(current);
    this._undoStack = { ...this._undoStack, [pageId]: undo };

    const next = stack.pop()!;
    this._redoStack = { ...this._redoStack, [pageId]: stack };
    this.applyUndoSnapshot(pageId, next);
    return this._materializedTables[pageId] ?? null;
  }

  // ── Save ──

  async save(pageId: string): Promise<Table> {
    const base = this._serverTables[pageId];
    if (!base) throw new Error("No table to save");

    // Rebuild materialized table to reflect all field overlays before delta extraction
    this.rematerialize(pageId);

    const { fields, appended, deleted, deletedIds } = this.getOverlays(pageId);
    const dirtyBaseIndices = [...fields.keys()].filter((i) => !deleted.has(i));
    const hasChanges = dirtyBaseIndices.length > 0 || appended.length > 0 || deletedIds.size > 0;

    let ipc: Uint8Array;
    let method: "POST" | "PATCH";

    if (hasChanges && (dirtyBaseIndices.length > 0 || appended.length > 0)) {
      const materialized = this._materializedTables[pageId];
      if (!materialized) throw new Error("No materialized table");

      const matBaseCount = base.numRows - deleted.size;
      const matIndices: number[] = [];
      for (const baseIdx of dirtyBaseIndices) {
        let matIdx = baseIdx;
        for (const d of deleted) {
          if (d < baseIdx) matIdx--;
        }
        matIndices.push(matIdx);
      }
      for (let a = 0; a < appended.length; a++) {
        matIndices.push(matBaseCount + a);
      }

      const delta = extractRows(materialized, matIndices);
      ipc = tableToIPC(delta, "stream");
      method = "PATCH";
    } else if (deletedIds.size > 0) {
      ipc = new Uint8Array(0);
      method = "PATCH";
    } else {
      const materialized = this._materializedTables[pageId];
      if (!materialized) throw new Error("No materialized table");
      ipc = tableToIPC(materialized, "stream");
      method = "POST";
    }

    // Persist via the injected transport — the engine no longer knows about HTTP
    // or routes. A conflict surfaces as AnnotationConflictError (thrown by the
    // transport) and propagates to the caller.
    const result = await this.transport.save(pageId, ipc, {
      method,
      ifMatch: this._versions[pageId] ?? null,
      deletedIds: [...deletedIds],
    });

    this._versions = { ...this._versions, [pageId]: result.etag };

    const fresh = tableFromIPC(result.bytes);
    this._serverTables = { ...this._serverTables, [pageId]: fresh };
    this._materializedTables = {
      ...this._materializedTables,
      [pageId]: fresh,
    };
    this.clearOverlays(pageId);
    return fresh;
  }
}

function concatBytes(chunks: Uint8Array[], totalLen: number): Uint8Array {
  const result = new Uint8Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}
