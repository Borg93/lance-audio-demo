/**
 * The labeling abstraction — mode-AGNOSTIC, so the annotator is never coupled to a
 * single "review queue" flow. Four orthogonal axes over the provenance-rich
 * `annotations` schema (which stays mode-blind: it records `source` + `status` +
 * `confidence`, never "manual" vs "bulk"):
 *
 *   Selection (target)  ·  Producer (who)  ·  Op (what)  ·  Execution (where)
 *
 * The three annotation MODES are regions in (Producer × Execution):
 *   - Manual        = human × interactive × {one, picked}          (Label Studio)
 *   - AI-assisted   = model|propagate × interactive × {one,picked,all-from-small}
 *                     (X-AnyLabeling · INSID3 few-shot · SAM click-to-segment)
 *   - Bulk / judge  = model|judge|propagate × batch × {query, all}
 *                     (ActiveLabelingSystem loop · FiftyOne embedding-interaction)
 *
 * The two VIEWER RESOLUTIONS are just how a Selection is formed — browse
 * (page/gallery/table → one|picked) vs search/filter/embedding-interact (→ query).
 * Both feed the SAME LabelOp. See docs/ACTIVE_LABELING.md.
 */

/** The target set of a labeling op. HOW it's formed (browse vs search) is the
 *  viewer's concern; the op only sees the resolved target. `query`/`all` are
 *  corpus-scale ⇒ batch execution; `one`/`picked` can run interactively. */
export type Selection =
  | { kind: "one"; index: number }
  | { kind: "picked"; indices: number[] }
  | {
      kind: "query";
      /** SQL predicate over the annotations/media columns (the FiftyOne filter). */
      where?: string;
      /** Embedding-interaction: nearest-k to an anchor vector in a column. */
      vector?: { column: string; anchor: number[]; k: number };
    }
  | { kind: "all" };

/** Who assigns the label — and the provenance stamp it writes to `annotations.source`. */
export type ProducerKind = "human" | "model" | "propagate" | "judge";

/** What happens to the target. */
export type Op = "set" | "verdict" | "predict" | "propagate" | "judge";

/** Where the op runs. Interactive = local-first overlay → Save (merge_insert);
 *  batch = a silver deriver / lance-ray job (async, replace-protects-humans). */
export type Execution = "interactive" | "batch";

/** One labeling operation — the single verb spanning all three modes. */
export interface LabelOp {
  target: Selection;
  /** Registry key of the producer (see producers.ts). */
  producer: string;
  op: Op;
  execution: Execution;
  payload: {
    /** set / verdict — the field changes (e.g. {status:"accepted"}). */
    fields?: Record<string, string>;
    /** predict — open-vocab / VLM prompt (grounding-dino text, HTR none). */
    prompt?: string;
    /** propagate — annotation indices used as the few-shot reference (INSID3). */
    exemplars?: number[];
  };
}

/** The mode-blind result of applying an op — annotation field deltas keyed by row
 *  index (interactive) or id (batch merge). Status/source/confidence come from the
 *  producer, not the caller. */
export interface LabelDelta {
  index?: number;
  id?: string;
  fields: Record<string, string | number>;
}

/** Outcome of dispatching a LabelOp. Interactive ops apply immediately (deltas land
 *  in the overlay); batch ops are enqueued and surface asynchronously by media id +
 *  Lance version. */
export type LabelOutcome =
  | { status: "applied"; deltas: LabelDelta[] }
  | { status: "queued"; job: string; note: string }
  | { status: "unsupported"; reason: string };
