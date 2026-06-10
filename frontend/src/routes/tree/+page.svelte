<script lang="ts">
  /**
   * Tree tab — the topic hierarchy as a zoomable treemap (or Sankey flow).
   *
   * `raudio feature topics` clusters chunks (Toponymy) and stores the nested
   * hierarchy as Lance JSONB in `topics.lance`; `/api/topics` serves it. We gate
   * on `built` (pointing the user at the build step otherwise) and render
   * <TopicTreemap>. Clicking a topic opens its results in a side panel
   * (<TopicResultsPanel>), playable in place; "Open in Search" continues to
   * `/?topic=<name>` for refining.
   *
   * A dedicated route — separate from Search and Atlas, sharing only the topic
   * filter on `/api/search`.
   */
  import { browser } from '$app/environment';
  import { getTopics, type TopicNode } from '$lib/api';
  import TopicTreemap from '$lib/components/topic-treemap.svelte';
  import TopicResultsPanel from '$lib/components/topic-results-panel.svelte';

  type Phase = 'loading' | 'ready' | 'unavailable' | 'error';

  let phase = $state<Phase>('loading');
  let errorMsg = $state<string | null>(null);
  let hierarchy = $state<TopicNode | null>(null);
  let noiseLabel = $state('');
  /** Topic whose results show in the side panel (null = full-width map). */
  let selectedTopic = $state<string | null>(null);

  $effect(() => {
    if (!browser) return;
    let cancelled = false;

    (async () => {
      try {
        const res = await getTopics();
        if (cancelled) return;
        if (!res.built || !res.hierarchy) {
          phase = 'unavailable';
          return;
        }
        hierarchy = res.hierarchy;
        noiseLabel = res.noise_label ?? ''; // '' (no match) on an older backend
        phase = 'ready';
      } catch (e) {
        if (!cancelled) {
          errorMsg = e instanceof Error ? e.message : String(e);
          phase = 'error';
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  });
</script>

<div class="h-full w-full">
  {#if phase === 'unavailable'}
    <div class="grid h-full place-items-center p-6 text-center text-sm text-muted-foreground">
      <div>
        <p class="mb-1 font-medium text-foreground">No topics yet</p>
        <p>
          Run <code class="rounded bg-muted px-1 py-0.5">raudio feature topics</code> to cluster the chunks
          into a topic hierarchy.
        </p>
      </div>
    </div>
  {:else if phase === 'error'}
    <div class="grid h-full place-items-center p-6 text-center text-sm text-destructive">
      Failed to load topics: {errorMsg}
    </div>
  {:else if phase === 'ready' && hierarchy}
    <!-- One always-mounted treemap (so drill/view state survives opening the
         panel) + a fixed-width results panel when a topic is clicked. -->
    <div class="flex h-full min-h-0">
      <div class="min-w-0 flex-1">
        <TopicTreemap {hierarchy} {noiseLabel} onShowResults={(name) => (selectedTopic = name)} />
      </div>
      {#if selectedTopic}
        <div class="w-96 shrink-0">
          <TopicResultsPanel topic={selectedTopic} onClose={() => (selectedTopic = null)} />
        </div>
      {/if}
    </div>
  {:else}
    <div class="grid h-full place-items-center text-sm text-muted-foreground">Loading…</div>
  {/if}
</div>
