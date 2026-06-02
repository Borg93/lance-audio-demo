<script lang="ts">
  /** Presentational flow diagram: how the search legs combine into one ranking. */
</script>

<div class="flex flex-col gap-2 text-[11px]">
  <p class="text-muted-foreground">
    Every result is one <strong class="text-foreground">chunk</strong> — a transcript span plus its
    video frame. A search runs up to three legs in parallel, then fuses them into a single ranking.
  </p>

  <!-- Inputs -->
  <div class="grid grid-cols-3 gap-2">
    <div class="rounded border border-border border-l-2 border-l-sky-400 bg-surface2 px-2 py-1.5 text-center">
      ⌨ Keyword<div class="text-[10px] text-muted-foreground">exact words</div>
    </div>
    <div class="rounded border border-border border-l-2 border-l-primary bg-surface2 px-2 py-1.5 text-center">
      💬 Meaning<div class="text-[10px] text-muted-foreground">your text</div>
    </div>
    <div class="rounded border border-border border-l-2 border-l-amber-400 bg-surface2 px-2 py-1.5 text-center">
      🖼 Image<div class="text-[10px] text-muted-foreground">uploaded</div>
    </div>
  </div>
  <div class="grid grid-cols-3 text-center text-muted-foreground/70">
    <span>▼</span><span>▼ embed</span><span>▼ embed</span>
  </div>

  <!-- Retrieval legs -->
  <div class="grid grid-cols-3 gap-2">
    <div class="rounded border border-border bg-background px-2 py-1.5 text-center">
      FTS<div class="text-[10px] text-muted-foreground">BM25 on text</div>
    </div>
    <div class="rounded border border-border bg-background px-2 py-1.5 text-center">
      Text vector<div class="text-[10px] text-muted-foreground">text_embedding</div>
    </div>
    <div class="rounded border border-border bg-background px-2 py-1.5 text-center">
      Frame vector<div class="text-[10px] text-muted-foreground">frame_embedding</div>
    </div>
  </div>
  <div class="text-center text-muted-foreground/70">▼&emsp;&emsp;▼&emsp;&emsp;▼</div>

  <!-- Fuse → rerank → results -->
  <div class="rounded-md border border-primary/50 bg-primary/10 px-3 py-2 text-center font-medium text-foreground">
    RRF fusion — merge by chunk (equal weight)
  </div>
  <div class="text-center text-muted-foreground/70">▼</div>
  <div class="rounded border border-border bg-surface2 px-3 py-1.5 text-center">
    Rerank <span class="text-muted-foreground">(optional)</span> — cross-encoder re-scores top candidates on the <em>text</em>
  </div>
  <div class="text-center text-muted-foreground/70">▼</div>
  <div class="rounded-md border border-border bg-card px-3 py-2 text-center font-semibold text-foreground">
    Ranked chunk results
  </div>

  <ul class="mt-1 list-disc space-y-1 pl-4 text-muted-foreground">
    <li><strong class="text-foreground">One merged hit per chunk</strong> — a chunk matched by several legs scores higher, not as separate rows.</li>
    <li><strong class="text-foreground">Cross-modal:</strong> text and image share one embedding space, so an image can match transcript meaning and text can match frames.</li>
    <li><strong class="text-foreground">Balance slider</strong> weights keyword ↔ vector in <em>Hybrid</em>; the 3-way image+text fusion is equal-weight RRF.</li>
    <li><strong class="text-foreground">Caveat:</strong> the image leg is visual frame similarity, not face-ID, and there's no speaker attribution linking who is shown to who is speaking.</li>
  </ul>
</div>
