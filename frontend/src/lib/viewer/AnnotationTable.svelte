<script lang="ts">
  // Table view of a unit's annotations — the REVIEW QUEUE. Reads the same Arrow IPC
  // the canvas does, and sorts the active-learning way (predictions first, by
  // uncertainty desc) so the shakiest model output is at the top to judge.
  import { tableFromIPC } from 'apache-arrow';
  import type { MediaUnit } from './types';

  let { unit, onselect }: { unit: MediaUnit; onselect?: (id: string) => void } = $props();

  type Row = {
    id: string;
    label: string;
    status: string;
    confidence: number;
    uncertainty: number;
    text: string;
    shape_type: string;
    source: string;
  };

  let rows = $state<Row[]>([]);
  let error = $state('');

  $effect(() => {
    let alive = true;
    fetch(unit.annotationsUrl)
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const t = tableFromIPC(new Uint8Array(await r.arrayBuffer()));
        const s = (n: string, i: number): string => String(t.getChild(n)?.get(i) ?? '');
        const num = (n: string, i: number): number => Number(t.getChild(n)?.get(i) ?? 0);
        const out: Row[] = [];
        for (let i = 0; i < t.numRows; i++) {
          out.push({
            id: s('id', i),
            label: s('label', i),
            status: s('status', i),
            confidence: num('confidence', i),
            uncertainty: num('uncertainty', i),
            text: s('text', i),
            shape_type: s('shape_type', i),
            source: s('source', i),
          });
        }
        // Review queue: predictions before humans, then most-uncertain first.
        out.sort((a, b) => {
          const pa = a.status === 'prediction' ? 0 : 1;
          const pb = b.status === 'prediction' ? 0 : 1;
          return pa - pb || b.uncertainty - a.uncertainty;
        });
        if (alive) rows = out;
      })
      .catch((e) => alive && (error = e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  });

  const statusColor = (s: string): string =>
    s === 'prediction'
      ? 'text-amber-600'
      : s === 'accepted'
        ? 'text-emerald-600'
        : s === 'rejected'
          ? 'text-red-600'
          : 'text-muted-foreground';
</script>

<div class="flex h-full flex-col overflow-hidden text-xs" data-testid="annotation-table">
  <div class="border-b border-border px-3 py-2 font-medium">
    Review queue · {rows.length} annotation{rows.length === 1 ? '' : 's'}
  </div>
  {#if error}
    <div class="p-3 text-destructive">failed: {error}</div>
  {:else}
    <div class="overflow-auto">
      <table class="w-full border-collapse">
        <thead class="sticky top-0 bg-muted/60 text-left text-[10px] uppercase text-muted-foreground">
          <tr>
            <th class="px-3 py-1.5">Label</th>
            <th class="px-2 py-1.5">Shape</th>
            <th class="px-2 py-1.5">Status</th>
            <th class="px-2 py-1.5 text-right">Conf</th>
            <th class="px-2 py-1.5 text-right">Uncert</th>
            <th class="px-3 py-1.5">Text</th>
          </tr>
        </thead>
        <tbody>
          {#each rows as row (row.id)}
            <tr
              class="cursor-pointer border-b border-border/50 hover:bg-muted/40"
              onclick={() => onselect?.(row.id)}
            >
              <td class="px-3 py-1.5">{row.label || '—'}</td>
              <td class="px-2 py-1.5 text-muted-foreground">{row.shape_type}</td>
              <td class="px-2 py-1.5 font-medium {statusColor(row.status)}">{row.status}</td>
              <td class="px-2 py-1.5 text-right tabular-nums">{row.confidence.toFixed(2)}</td>
              <td class="px-2 py-1.5 text-right tabular-nums">{row.uncertainty.toFixed(2)}</td>
              <td class="max-w-[16rem] truncate px-3 py-1.5" title={row.text}>{row.text || '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>
