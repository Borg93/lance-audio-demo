<script lang="ts">
  import { Input } from '$lib/components/ui';
  import { Popover } from 'bits-ui';
  import type { SearchSpec } from '$lib/api';
  import { Filter, X } from 'lucide-svelte';
  import { cn } from '$lib/utils';

  type Props = {
    spec: SearchSpec;
    onchange?: () => void;
  };
  let { spec = $bindable(), onchange }: Props = $props();

  let language = $state(spec.language ?? '');
  let namn = $state(spec.namn ?? '');
  let referenskod = $state(spec.referenskod ?? '');
  let extraid = $state(spec.extraid ?? '');

  function commit() {
    spec = {
      ...spec,
      language: language || undefined,
      namn: namn || undefined,
      referenskod: referenskod || undefined,
      extraid: extraid || undefined,
    };
    onchange?.();
  }

  function clear() {
    language = '';
    namn = '';
    referenskod = '';
    extraid = '';
    commit();
  }

  const activeCount = $derived(
    [language, namn, referenskod, extraid].filter(Boolean).length,
  );
</script>

<Popover.Root>
  <Popover.Trigger
    class={cn(
      'inline-flex h-9 items-center gap-1.5 rounded-md border border-border bg-input px-3 text-xs',
      activeCount > 0 ? 'text-foreground' : 'text-muted-foreground',
      'hover:text-foreground',
    )}
    title="Filter results by language, event name, reference code, or internal ID"
  >
    <Filter class="size-3.5" />
    <span>Filters</span>
    {#if activeCount > 0}
      <span class="ml-1 rounded-full bg-primary px-1.5 text-[10px] font-bold text-primary-foreground">
        {activeCount}
      </span>
    {/if}
  </Popover.Trigger>

  <Popover.Portal>
    <Popover.Content
      sideOffset={6}
      align="end"
      class="z-50 w-[320px] rounded-md border border-border bg-card p-3 text-xs shadow-md"
    >
      <div class="mb-2 flex items-center">
        <span class="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Filter results
        </span>
        {#if activeCount > 0}
          <button
            type="button"
            onclick={clear}
            class="ml-auto flex items-center gap-0.5 text-[11px] text-muted-foreground hover:text-foreground"
          >
            <X class="size-3" /> Clear
          </button>
        {/if}
      </div>

      <div class="grid gap-2">
        <label class="flex flex-col gap-1">
          <span class="text-muted-foreground">Language</span>
          <select
            bind:value={language}
            onchange={commit}
            class="h-8 rounded-md border border-border bg-input px-2 text-xs text-foreground"
          >
            <option value="">Any</option>
            <option value="sv">Swedish</option>
            <option value="en">English</option>
          </select>
        </label>

        <label class="flex flex-col gap-1">
          <span class="text-muted-foreground">
            Event name <span class="text-muted-foreground/70">(contains)</span>
          </span>
          <Input bind:value={namn} onchange={commit} placeholder="e.g. alkoholmonopolet" class="h-8 text-xs" />
        </label>

        <label class="flex flex-col gap-1">
          <span class="text-muted-foreground">
            Reference code <span class="text-muted-foreground/70">(contains)</span>
          </span>
          <Input bind:value={referenskod} onchange={commit} placeholder="e.g. SE/RA/1207" class="h-8 text-xs" />
        </label>

        <label class="flex flex-col gap-1">
          <span class="text-muted-foreground">
            Internal ID <span class="text-muted-foreground/70">(exact)</span>
          </span>
          <Input bind:value={extraid} onchange={commit} placeholder="e.g. T0001347" class="h-8 text-xs" />
        </label>
      </div>
    </Popover.Content>
  </Popover.Portal>
</Popover.Root>
