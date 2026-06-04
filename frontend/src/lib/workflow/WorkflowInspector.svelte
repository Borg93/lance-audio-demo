<script lang="ts">
  /** Persistent right panel. Click a node → see its inputs + intermediate
   *  results; click a result → play it here (reuses PlayerPane). */
  import { ArrowLeft } from 'lucide-svelte';
  import { graph, modeLabel, nodeLabel, STATUS_DOT } from '$lib/workflow/graph.svelte';
  import PlayerPane from '$lib/components/player-pane.svelte';
  import HitList from '$lib/workflow/HitList.svelte';

  const id = $derived(graph.inspectedNodeId);
  const kind = $derived(id ? graph.kindOf(id) : null);
  const cfg = $derived(id ? graph.config[id] : null);
  const rt = $derived(id ? graph.runtime[id] : null);
  const hits = $derived(rt?.hits ?? []);

  const title = $derived(
    kind === 'search' && cfg ? `Search · ${modeLabel(cfg.mode)}` : kind ? nodeLabel(kind) : '',
  );

  const statusText = $derived.by((): string => {
    if (!rt) return '';
    if (rt.status === 'running') return 'searching…';
    if (rt.status === 'error') return 'error';
    if (rt.status === 'done') return rt.count != null ? `done · ${rt.count} hits` : 'done';
    return 'not run yet';
  });

  // Per-kind input/config summary (only the fields that matter for that kind).
  const rows = $derived.by((): [string, string][] => {
    if (!cfg || !kind) return [];
    const r: [string, string][] = [];
    if (kind === 'query') r.push(['Query', cfg.q || '—']);
    if (kind === 'image') r.push(['Image', cfg.imageName || '(none uploaded)']);
    if (kind === 'filter') {
      if (cfg.where) r.push(['Where', cfg.where]);
      if (cfg.language) r.push(['Language', cfg.language]);
      if (cfg.namn) r.push(['Name', cfg.namn]);
      if (!r.length) r.push(['Filter', '(empty)']);
    }
    if (kind === 'search') {
      r.push(['Mode', modeLabel(cfg.mode)]);
      r.push(['Query', cfg.q || (cfg.mode === 'visual' ? '(from image)' : '—')]);
      r.push(['Results', String(cfg.n)]);
      if (cfg.rerank) r.push(['Rerank', 'top 20']);
      if (rt?.scopedDocs) r.push(['Scope', `within ${rt.scopedDocs} videos`]);
      if (rt?.ms != null) r.push(['Time', `${rt.ms} ms`]);
    }
    return r;
  });
</script>

<div data-testid="inspector" class="flex h-full min-h-0 flex-col border-l border-border bg-card">
  <header class="flex h-11 shrink-0 items-center gap-2 border-b border-border px-3">
    {#if graph.selectedHit}
      <button
        type="button"
        onclick={() => graph.closeDetail()}
        aria-label="Back to results"
        class="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      >
        <ArrowLeft class="size-4" />
      </button>
      <span class="truncate text-sm font-medium text-foreground">
        {graph.selectedHit.namn || 'Now playing'}
      </span>
    {:else}
      <span class="text-sm font-medium text-foreground">Inspector</span>
      {#if title}<span class="truncate text-xs text-muted-foreground">· {title}</span>{/if}
    {/if}
  </header>

  <div class="min-h-0 flex-1 overflow-y-auto">
    {#if graph.selectedHit}
      <PlayerPane hit={graph.selectedHit} />
    {:else if id && kind && cfg && rt}
      <div class="flex flex-col gap-3 p-3 text-xs">
        <div class="flex items-center gap-2">
          <span class="size-2 shrink-0 rounded-full {STATUS_DOT[rt.status]}"></span>
          <span class="text-muted-foreground">{statusText}</span>
        </div>
        {#if rt.error}
          <div class="rounded border border-destructive/30 bg-destructive/10 p-2 text-destructive">
            {rt.error}
          </div>
        {/if}

        <dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
          {#each rows as [label, value] (label)}
            <dt class="text-muted-foreground">{label}</dt>
            <dd class="break-words font-medium text-foreground">{value}</dd>
          {/each}
        </dl>

        {#if hits.length}
          <div>
            <div class="mb-1 text-[10px] tracking-wide text-muted-foreground uppercase">
              Results ({hits.length}) · click to play
            </div>
            <HitList {hits} maxHeight="max-h-none" />
          </div>
        {:else if kind === 'search' || kind === 'results'}
          <p class="text-[11px] text-muted-foreground">
            {rt.status === 'idle' ? 'Not run yet — press Run.' : 'No results.'}
          </p>
        {:else}
          <p class="text-[11px] text-muted-foreground">
            Produces a {nodeLabel(kind).toLowerCase()} input — wire it into a Search and Run.
          </p>
        {/if}
      </div>
    {:else}
      <div class="grid h-full place-items-center p-6 text-center text-xs text-muted-foreground">
        Click a node to inspect its inputs &amp; results — or click a result to play it.
      </div>
    {/if}
  </div>
</div>
