<script lang="ts">
  /**
   * Tree tab — the topic hierarchy as a zoomable treemap.
   *
   * `raudio feature topics` clusters chunks (Toponymy) and stores the nested
   * hierarchy as Lance JSONB in `topics.lance`; `/api/topics` serves it. We gate
   * on `built` (pointing the user at the build step otherwise) and render
   * <TopicTreemap>. Clicking a topic hands off to Search via `/?topic=<name>`.
   *
   * A dedicated route — separate from Search and Atlas, sharing only the topic
   * filter on `/api/search`.
   */
  import { browser } from '$app/environment';
  import { getTopics, type TopicNode } from '$lib/api';
  import TopicTreemap from '$lib/components/topic-treemap.svelte';

  type Phase = 'loading' | 'ready' | 'unavailable' | 'error';

  let phase = $state<Phase>('loading');
  let errorMsg = $state<string | null>(null);
  let hierarchy = $state<TopicNode | null>(null);

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
          Run <code class="rounded bg-muted px-1 py-0.5">raudio feature topics</code> to cluster the
          chunks into a topic hierarchy.
        </p>
      </div>
    </div>
  {:else if phase === 'error'}
    <div class="grid h-full place-items-center p-6 text-center text-sm text-destructive">
      Failed to load topics: {errorMsg}
    </div>
  {:else if phase === 'ready' && hierarchy}
    <TopicTreemap {hierarchy} />
  {:else}
    <div class="grid h-full place-items-center text-sm text-muted-foreground">Loading…</div>
  {/if}
</div>
