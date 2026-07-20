// Persistence transport — lets the framework-agnostic AnnotationStore save/load
// WITHOUT knowing about HTTP, routes, or the host app. The Svelte binding injects
// a concrete implementation (see src/lib/stores/httpAnnotationTransport.ts); a
// test or a different host can inject its own (in-memory, IndexedDB, gRPC, …).

export interface AnnotationLoadResult {
  /** A progressive byte stream for incremental parse, or null if only `bytes`. */
  stream: ReadableStream<Uint8Array> | null;
  /** Full IPC bytes, or null if a `stream` is provided instead. */
  bytes: Uint8Array | null;
  /** Opaque version tag (e.g. an ETag) for optimistic concurrency. */
  etag: string;
}

export interface AnnotationSaveContext {
  /** "POST" = full table overwrite, "PATCH" = delta. */
  method: "POST" | "PATCH";
  /** The version the client loaded, for If-Match-style optimistic concurrency. */
  ifMatch: string | null;
  /** Annotation ids deleted since load. */
  deletedIds: string[];
}

export interface AnnotationSaveResult {
  /** The authoritative stored bytes (server's view after the write). */
  bytes: Uint8Array;
  /** The new version tag. */
  etag: string;
}

export interface AnnotationTransport {
  /** Load stored IPC for a page, or null if none exists yet. */
  load(pageId: string): Promise<AnnotationLoadResult | null>;
  /** Persist IPC; return the authoritative stored bytes + new version. */
  save(pageId: string, ipc: Uint8Array, ctx: AnnotationSaveContext): Promise<AnnotationSaveResult>;
}

/** Thrown by a transport when the backend reports a version conflict (e.g. HTTP 409). */
export class AnnotationConflictError extends Error {
  constructor(message = "Conflict: data was modified elsewhere. Please reload.") {
    super(message);
    this.name = "AnnotationConflictError";
  }
}
