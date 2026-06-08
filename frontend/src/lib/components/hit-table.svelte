<script lang="ts" module>
  import type { Hit } from '$lib/api';
  import { fmtTime } from '$lib/utils';

  export type TableColumn = { key: string; label: string; render: (h: Hit) => string };

  /** Result fields shown as table columns (everything the search payload carries
   *  — see api.ts:HitSchema). `thumbnail` and `text` are rendered specially in
   *  the markup (image / highlighted HTML); their `render` is unused. */
  export const TABLE_COLUMNS: TableColumn[] = [
    { key: 'thumbnail', label: 'Thumb', render: () => '' },
    { key: 'score', label: 'Score', render: (h) => (h._score != null ? h._score.toFixed(3) : '') },
    { key: 'namn', label: 'Name', render: (h) => h.namn ?? '' },
    { key: 'referenskod', label: 'Ref', render: (h) => h.referenskod ?? '' },
    { key: 'bildid', label: 'Bild ID', render: (h) => h.bildid ?? '' },
    { key: 'extraid', label: 'Internal ID', render: (h) => h.extraid ?? '' },
    { key: 'language', label: 'Lang', render: (h) => h.language ?? '' },
    { key: 'speech_id', label: 'Speech', render: (h) => String(h.speech_id) },
    { key: 'chunk_id', label: 'Chunk', render: (h) => String(h.chunk_id) },
    { key: 'start', label: 'Start', render: (h) => fmtTime(h.start) },
    { key: 'end', label: 'End', render: (h) => fmtTime(h.end) },
    { key: 'duration', label: 'Dur', render: (h) => (h.duration != null ? fmtTime(h.duration) : '') },
    { key: 'doc_id', label: 'Doc', render: (h) => h.doc_id },
    { key: 'audio_path', label: 'File', render: (h) => h.audio_path },
    { key: 'text', label: 'Text', render: (h) => h.text },
    { key: 'caption', label: 'Caption', render: (h) => h.caption ?? '' },
  ];
</script>

<script lang="ts">
  import { thumbnailUrl } from '$lib/api';
  import { queryTerms, makeHighlighter, hitKey } from '$lib/utils';

  let {
    hits,
    active,
    onselect,
    visible,
    query = '',
  }: {
    hits: Hit[];
    active: Hit | null;
    onselect?: (h: Hit) => void;
    visible: string[];
    query?: string;
  } = $props();

  const cols = $derived(TABLE_COLUMNS.filter((c) => visible.includes(c.key)));
  const terms = $derived(queryTerms(query));
  const highlight = $derived(makeHighlighter(terms));
  const activeKey = $derived(active ? hitKey(active) : null);
</script>

<div class="overflow-x-auto">
  <table class="w-full border-collapse text-xs">
    <thead>
      <tr class="sticky top-0 z-10 border-b border-border bg-card text-left text-muted-foreground">
        {#each cols as c (c.key)}
          <th class="px-3 py-2 font-medium whitespace-nowrap">{c.label}</th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each hits as hit (hitKey(hit))}
        <tr
          class={'cursor-pointer border-b border-border/60 hover:bg-secondary/40 ' +
            (activeKey === hitKey(hit)
              ? 'bg-primary/15 font-medium [box-shadow:inset_3px_0_0_0_var(--color-primary)]'
              : '')}
          onclick={() => onselect?.(hit)}
        >
          {#each cols as c (c.key)}
            {#if c.key === 'thumbnail'}
              <td class="px-3 py-1.5 align-top">
                <img
                  src={thumbnailUrl(hit.doc_id)}
                  loading="lazy"
                  alt=""
                  class="h-9 w-16 rounded bg-muted object-cover"
                  onerror={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = 'hidden')}
                />
              </td>
            {:else if c.key === 'text'}
              <td class="max-w-[32rem] px-3 py-1.5 align-top text-foreground [overflow-wrap:anywhere]" title={hit.text}>
                <!-- highlight() escapes then wraps matches — safe to inject -->
                <div class="line-clamp-2">{@html highlight(hit.text)}</div>
              </td>
            {:else}
              <td
                class="max-w-[28rem] truncate px-3 py-1.5 align-top whitespace-nowrap text-muted-foreground"
                title={c.render(hit)}
              >
                {c.render(hit)}
              </td>
            {/if}
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>
