<script lang="ts">
  // The annotator shell — decoupled from any specific viewer. It resolves a
  // MediaUnit and renders whatever viewer the registry maps its kind to (image =
  // PixiJS canvas today; audio/video = scaffolded stubs). Adding a modality never
  // touches this route. (RA_ANNO_MERGE.md §5c–5d.)
  import { mediaKindOf, viewerFor } from '$lib/viewer/registry';
  import type { MediaUnit } from '$lib/viewer/types';

  // Demo unit: the seeded image doc. In the app this comes from the descriptor
  // (`document.mime` → kind) + the selected doc/frame — same shape, resolved upstream.
  const KEY = 'fe00cd746463ad2c/0/19';
  const unit: MediaUnit = {
    kind: mediaKindOf('image/jpeg'),
    key: KEY,
    imageUrl: `/api/chunk-frame/${KEY}`,
    annotationsUrl: `/api/annotations/${KEY}`,
  };
  const Viewer = viewerFor(unit.kind);

  let status = $state('loading…');
</script>

<div class="relative h-screen w-screen">
  <div
    class="absolute left-3 top-3 z-10 rounded bg-black/70 px-2 py-1 font-mono text-xs text-white"
    data-testid="annotate-status"
  >
    annotate · {unit.kind} · {status}
  </div>
  <Viewer {unit} onload={(n) => (status = `${n} annotations from Lance`)} />
</div>
