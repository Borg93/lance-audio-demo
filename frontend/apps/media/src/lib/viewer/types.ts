/**
 * The viewer/annotation ABSTRACTION — the seam that keeps modalities decoupled.
 *
 * The only axis that forks per media kind is the VIEWER component (image = PixiJS
 * canvas, audio = waveform, video = frame-overlay + timeline). Everything else —
 * the annotation model, the store, the annotator service — stays uniform. Adding a
 * modality = one registry entry + one component that satisfies `ViewerProps`, not a
 * rewire. (See RA_ANNO_MERGE.md §5c–5d.)
 */

/** Media modality of a unit to annotate — the sole per-modality fork point. */
export type MediaKind = "image" | "audio" | "video";

/** Everything a viewer needs to render + annotate one unit, resolved from the
 *  descriptor upstream so the viewer itself is decoupled from routing/data. */
export interface MediaUnit {
  kind: MediaKind;
  /** Stable id (doc + identity keys joined) — the annotation key scope. */
  key: string;
  /** Backdrop image URL: the page image (image) or the current frame (video). */
  imageUrl?: string;
  /** Time-media URL: the `<audio>`/`<video>` source (audio, video). */
  mediaUrl?: string;
  /** Arrow-IPC endpoint for this unit's annotations. */
  annotationsUrl: string;
  /** Natural media dimensions when known. */
  width?: number;
  height?: number;
}

/** The props EVERY viewer component accepts — the decoupling contract. A new
 *  modality's component takes exactly these, so the shell/route never special-cases. */
export interface ViewerProps {
  unit: MediaUnit;
  /** Fired once the viewer has loaded + rendered (`count` = annotations shown). */
  onload?: (count: number) => void;
  /** Optional route-level annotator facade. A spatial viewer `attach`es its engine
   *  + loaded table here so the ra-anno layout (toolbar/sidebar/zoom/layers) can bind
   *  to a single reactive source; temporal viewers may attach their own surface later.
   *  Omitted ⇒ the viewer renders read-only, exactly as before. */
  controller?: import("./annotator.svelte").AnnotatorController;
}

/** Spatial geometry — image + video-frame. Mirrors the engine's shape model
 *  (frontend/src/lib/engine/schema.ts): bbox, oriented box, polygon, HTR baseline,
 *  line, point, and raster MASK. */
export interface SpatialGeom {
  shape_type: "rectangle" | "rotation" | "polygon" | "baseline" | "line" | "point" | "mask";
  x: number;
  y: number;
  width: number;
  height: number;
  rotation?: number;
  /** Flat `[x0,y0,x1,y1,...]`. */
  polygon?: number[];
  /** Base64 PNG (data URL) for raster brush masks. */
  mask?: string;
}

/** Temporal facet — audio/video segments + a shape pinned to a video moment.
 *  OPTIONAL: spatial-only rows (documents/HTR) omit it entirely, so ONE annotation
 *  model spans all modalities without a fork. */
export interface TemporalFacet {
  /** Segment start/end, seconds (audio + video timeline). */
  t_start?: number;
  t_end?: number;
  /** A spatial shape pinned to a specific video frame. */
  frame_idx?: number;
}

/** One annotation across ALL modalities — spatial facet + optional temporal facet
 *  + shared attributes. VIDEO = spatial (mask/bbox/polygon over a frame) AND temporal
 *  (frame_idx/t); AUDIO = temporal only; IMAGE = spatial only. Same table, same
 *  service — the geometry present says which modality it belongs to. */
export type Annotation = Partial<SpatialGeom> &
  TemporalFacet & {
    id: string;
    label?: string;
    status?: string;
    group_id?: string;
    text?: string;
  };
