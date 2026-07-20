// Arrow schema definitions — shared between server and client

export const ANNOTATION_COLUMNS = [
  "id",
  "page_id",
  "dataset_id",
  "shape_type", // Utf8 — geometry kind, see ShapeType
  "x",
  "y",
  "width",
  "height",
  "rotation", // Float32 — oriented-box angle in radians (0 = axis-aligned)
  "polygon", // List<Float32> — flat [x0,y0,x1,y1,...]; OPEN ring when shape_type is 'baseline'/'line'
  "text", // transcription / content
  "label", // class / region type (free-text, or from REGION_TYPES)
  "confidence", // model score 0..1
  "source",
  "status",
  "reviewer",
  "group", // layer/category grouping (free-text)
  "group_id", // Utf8 — instance/linking key (groups split blocks, lines→region)
  "reading_order", // Int32 — reading sequence within page (-1 = unset)
  "difficult", // Bool — ambiguous/ignore (exclude from training)
  "links", // Utf8 JSON — AnnotationRelation[] to other annotations (kie_linking)
  "mask", // Utf8 — base64 PNG (data URL) for raster brush masks; "" when not a mask
  "metadata", // Utf8 JSON — arbitrary typed attributes
] as const;

export type AnnotationStatus = "prediction" | "draft" | "reviewed" | "accepted" | "rejected";

export type AnnotationSource = "manual" | `model:${string}`;

/**
 * Geometry kind. Unified model: every shape is rings of points + flags.
 * Axis-aligned/oriented boxes also store x/y/width/height(/rotation) for fast bbox ops;
 * polygon holds flat [x,y,...] coords. 'baseline'/'line' are OPEN rings (not closed).
 */
export type ShapeType =
  | "rectangle" // axis-aligned box
  | "rotation" // oriented box (uses rotation, radians)
  | "polygon" // closed ring
  | "baseline" // open polyline — the HTR text-line baseline
  | "line" // open 2+ point line
  | "point" // single point
  | "mask"; // raster mask (base64 in `mask` column) — from the brush tool

export const SHAPE_TYPES: readonly ShapeType[] = [
  "rectangle",
  "rotation",
  "polygon",
  "baseline",
  "line",
  "point",
  "mask",
] as const;

/** Relation/link between annotations (X-AnyLabeling kie_linking). Stored JSON-encoded in `links`. */
export interface AnnotationRelation {
  to: string; // target annotation id
  type: RelationType;
}

export type RelationType =
  | "key-value" // form key → value
  | "continued" // line/region continues in another
  | "reading-next" // explicit reading-order successor
  | "parent" // child → parent region (region→line→word)
  | "reference"; // marginalia/footnote → referent

/**
 * Suggested controlled region/line vocabulary (PAGE-XML / SegmOnto-flavored).
 * `label` may use these or stay free-text — NOT enforced by the schema.
 */
export const REGION_TYPES = [
  "paragraph",
  "heading",
  "text-line",
  "word",
  "marginalia",
  "page-number",
  "header",
  "footer",
  "table",
  "table-cell",
  "figure",
  "caption",
  "signature",
  "stamp-seal",
  "formula",
  "catch-word",
] as const;

export type RegionType = (typeof REGION_TYPES)[number];

/** Page-level table — one row per page, Binary columns for images */
export const PAGE_COLUMNS = [
  "page_id",
  "document_id",
  "dataset_id",
  "page_number",
  "image", // Binary — full resolution page image (JPEG/PNG/SVG)
  "thumbnail", // Binary — small preview (WebP/JPEG)
  "image_mime", // Utf8 — MIME type for the image
  "image_width",
  "image_height",
  "embedding", // FixedSizeList<Float32> — high-dimensional embedding vector
  "umap_x", // Float32 — 2D UMAP projection X
  "umap_y", // Float32 — 2D UMAP projection Y
] as const;
