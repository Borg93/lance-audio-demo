<script lang="ts">
  import type { SearchSpec } from '$lib/api';
  import { X } from 'lucide-svelte';

  type Props = {
    spec: SearchSpec;
    onchange?: (spec: SearchSpec) => void;
  };
  let { spec = $bindable(), onchange }: Props = $props();

  /** Visible filter pills with X buttons. Used to fix the "filter stuck
      invisibly behind a popover" bug. */
  type Pill = { key: keyof SearchSpec; label: string; value: string };
  const pills = $derived.by<Pill[]>(() => {
    const out: Pill[] = [];
    if (spec.language) out.push({ key: 'language', label: 'Language', value: spec.language });
    if (spec.namn) out.push({ key: 'namn', label: 'Name', value: spec.namn });
    if (spec.referenskod) out.push({ key: 'referenskod', label: 'Ref', value: spec.referenskod });
    if (spec.extraid) out.push({ key: 'extraid', label: 'ID', value: spec.extraid });
    if (spec.topic) out.push({ key: 'topic', label: 'Topic', value: spec.topic });
    if (spec.where) {
      const expr = spec.where.length > 48 ? `${spec.where.slice(0, 48)}…` : spec.where;
      out.push({ key: 'where', label: 'SQL', value: expr });
    }
    return out;
  });

  function clear(key: keyof SearchSpec) {
    spec = { ...spec, [key]: undefined };
    onchange?.(spec);
  }

  function clearAll() {
    spec = {
      ...spec,
      language: undefined,
      namn: undefined,
      referenskod: undefined,
      extraid: undefined,
      topic: undefined,
      where: undefined,
    };
    onchange?.(spec);
  }
</script>

{#if pills.length}
  <div class="flex flex-wrap items-center gap-1.5 px-6 pb-3 text-[11px]">
    <span class="text-muted-foreground">Active filters:</span>
    {#each pills as p (p.key)}
      <span class="flex items-center gap-1 rounded-full border border-border bg-secondary px-2 py-0.5 font-medium">
        <span class="text-muted-foreground">{p.label}:</span>
        <span class="max-w-[280px] truncate">{p.value}</span>
        <button
          type="button"
          aria-label="Remove {p.label} filter"
          onclick={() => clear(p.key)}
          class="text-muted-foreground hover:text-destructive"
        >
          <X class="size-3" />
        </button>
      </span>
    {/each}
    {#if pills.length > 1}
      <button
        type="button"
        onclick={clearAll}
        class="ml-1 text-muted-foreground hover:text-foreground"
      >
        Clear all
      </button>
    {/if}
  </div>
{/if}
