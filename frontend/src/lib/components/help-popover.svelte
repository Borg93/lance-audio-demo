<script lang="ts">
  import { Popover } from 'bits-ui';
  import { HelpCircle } from 'lucide-svelte';
  import SearchFlow from './search-flow.svelte';

  type Example = { label: string; example: string; explain: string };

  type Props = {
    examples: Record<string, Example>;
    /** Called when the user clicks an example row — fires (key, example). */
    onpick?: (key: string, example: string) => void;
  };
  let { examples, onpick }: Props = $props();

  let tab = $state<'examples' | 'flow'>('examples');
</script>

<Popover.Root>
  <Popover.Trigger
    class="flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground hover:bg-secondary/40 hover:text-foreground"
    title="Show search examples"
  >
    <HelpCircle class="size-4" />
  </Popover.Trigger>

  <Popover.Portal>
    <Popover.Content
      sideOffset={6}
      align="end"
      class="z-50 w-[520px] rounded-md border border-border bg-card p-3 text-xs shadow-md"
    >
      <!-- Tabs: examples vs the search-flow diagram -->
      <div class="mb-3 flex items-center gap-1 border-b border-border pb-2">
        <button
          type="button"
          onclick={() => (tab = 'examples')}
          class={'rounded px-2 py-0.5 text-[11px] font-medium transition-colors ' +
            (tab === 'examples' ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground')}
        >
          Examples
        </button>
        <button
          type="button"
          onclick={() => (tab = 'flow')}
          class={'rounded px-2 py-0.5 text-[11px] font-medium transition-colors ' +
            (tab === 'flow' ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground')}
        >
          How search works
        </button>
      </div>

      {#if tab === 'examples'}
        <div class="grid gap-1">
          {#each Object.entries(examples) as [key, info] (key)}
            <button
              type="button"
              onclick={() => onpick?.(key, info.example)}
              class="grid grid-cols-[140px_max-content_1fr] items-baseline gap-3 rounded px-2 py-1.5 text-left transition-colors hover:bg-secondary/50"
            >
              <span class="font-medium">{info.label}</span>
              <code class="rounded bg-surface2 px-1.5 py-0.5 font-mono text-[11px] text-primary">
                {info.example}
              </code>
              <span class="text-muted-foreground">{info.explain}</span>
            </button>
          {/each}
        </div>
        <div class="mt-3 rounded-md border border-dashed border-border bg-muted/40 p-2 text-[11px] text-muted-foreground">
          💡 <strong>With an image attached</strong> (drag in or 📎): the
          dropdown becomes a fusion knob.
          <ul class="mt-1 ml-4 list-disc space-y-0.5">
            <li><em>Keyword/Phrase/Fuzzy</em> → image-only (visual similarity).</li>
            <li><em>Vector</em> → image + your text query, fused over transcript&nbsp;+ frame vectors.</li>
            <li><em>Hybrid</em> → adds keyword search on top → 3-way RRF fusion.</li>
          </ul>
        </div>
      {:else}
        <SearchFlow />
      {/if}
    </Popover.Content>
  </Popover.Portal>
</Popover.Root>
