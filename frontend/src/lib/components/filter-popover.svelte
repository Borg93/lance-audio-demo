<script lang="ts">
  import { Input, Button, type SelectOption } from '$lib/components/ui';
  import { Popover } from 'bits-ui';
  import { listColumns, type SearchSpec, type ColumnInfo } from '$lib/api';
  import { Filter, X, Eye, EyeOff, Plus } from 'lucide-svelte';
  import { cn } from '$lib/utils';

  type Props = {
    spec: SearchSpec;
    onchange?: () => void;
  };
  let { spec = $bindable(), onchange }: Props = $props();

  // Language is a tiny enum, kept as a one-click convenience. Everything else
  // goes through the generic column builder below, which compiles to `where`.
  let language = $state(spec.language ?? '');
  let whereSql = $state(spec.where ?? '');

  function commit() {
    // Always prefilter (correct + uses the scalar index; never returns < n).
    spec = { ...spec, language: language || undefined, where: whereSql || undefined };
    onchange?.();
  }

  function clearAll() {
    language = '';
    whereSql = '';
    commit();
  }

  const activeCount = $derived([language, whereSql].filter(Boolean).length);

  // ── Filterable columns (from the backend) ──────────────────────────────
  let columns = $state<ColumnInfo[]>([]);
  $effect(() => {
    listColumns()
      .then((c) => (columns = c))
      .catch(() => (columns = []));
  });

  // Per-column hide, persisted so a decluttered list survives reloads.
  const HIDE_KEY = 'raudio-hidden-cols';
  let hidden = $state<Set<string>>(new Set());
  $effect(() => {
    try {
      const v = localStorage.getItem(HIDE_KEY);
      if (v) hidden = new Set(JSON.parse(v) as string[]);
    } catch {
      /* ignore */
    }
  });
  function toggleHide(name: string) {
    const next = new Set(hidden);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    hidden = next;
    try {
      localStorage.setItem(HIDE_KEY, JSON.stringify([...next]));
    } catch {
      /* ignore */
    }
  }

  const visibleCols = $derived(columns.filter((c) => !hidden.has(c.name)));

  // ── Filter builder: column · operator · value → predicate → WHERE ──────
  const NUMBER_OPS: SelectOption[] = [
    { value: '=', label: '=' },
    { value: '!=', label: '≠' },
    { value: '>', label: '>' },
    { value: '>=', label: '≥' },
    { value: '<', label: '<' },
    { value: '<=', label: '≤' },
  ];
  const TEXT_OPS: SelectOption[] = [
    { value: 'contains', label: 'contains' },
    { value: 'equals', label: 'equals' },
    { value: 'starts', label: 'starts with' },
  ];

  let colName = $state('');
  let op = $state('contains');
  let val = $state('');

  const colType = $derived(columns.find((c) => c.name === colName)?.type ?? 'text');
  const opOptions = $derived(colType === 'number' ? NUMBER_OPS : TEXT_OPS);
  // Keep the operator valid when the column type changes.
  $effect(() => {
    const first = opOptions[0];
    if (first && !opOptions.some((o) => o.value === op)) op = first.value;
  });

  function buildClause(): string | null {
    if (!colName || !val.trim()) return null;
    const v = val.trim();
    if (colType === 'number') {
      const num = Number(v);
      if (Number.isNaN(num)) return null;
      return `${colName} ${op} ${num}`;
    }
    const esc = v.replace(/'/g, "''");
    if (op === 'equals') return `${colName} = '${esc}'`;
    if (op === 'starts') return `${colName} LIKE '${esc}%'`;
    return `${colName} LIKE '%${esc}%'`;
  }

  function addFilter() {
    const clause = buildClause();
    if (!clause) return;
    whereSql = whereSql.trim() ? `${whereSql.trim()} AND ${clause}` : clause;
    val = '';
    commit(); // ← runs the search immediately so results visibly update
  }

  let manageOpen = $state(false);
</script>

<Popover.Root>
  <Popover.Trigger
    class={cn(
      'inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-background px-3 text-xs transition-colors',
      activeCount > 0 ? 'text-foreground' : 'text-muted-foreground',
      'hover:bg-muted hover:text-foreground data-[state=open]:bg-muted data-[state=open]:text-foreground',
    )}
    title="Filter results by any column"
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
      class="z-50 flex max-h-[75vh] w-[340px] flex-col gap-3 overflow-y-auto rounded-md border border-border bg-card p-3 text-xs shadow-md"
    >
      <div class="flex items-center">
        <span class="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Filter results
        </span>
        {#if activeCount > 0}
          <button
            type="button"
            onclick={clearAll}
            class="ml-auto flex items-center gap-0.5 text-[11px] text-muted-foreground hover:text-foreground"
          >
            <X class="size-3" /> Clear
          </button>
        {/if}
      </div>

      <!-- Builder: column · operator · value → Add -->
      <div class="flex flex-col gap-1.5">
        <span class="text-muted-foreground">Add a filter</span>
        <div class="flex gap-1.5">
          <select
            bind:value={colName}
            aria-label="Column"
            class="h-8 flex-1 rounded-md border border-border bg-background px-2 text-xs text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <option value="" disabled>Column…</option>
            {#each visibleCols as c (c.name)}
              <option value={c.name}>{c.name} · {c.type}</option>
            {/each}
          </select>
          <select
            bind:value={op}
            aria-label="Operator"
            class="h-8 w-28 rounded-md border border-border bg-background px-2 text-xs text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {#each opOptions as o (o.value)}
              <option value={o.value}>{o.label}</option>
            {/each}
          </select>
        </div>
        <div class="flex gap-1.5">
          <Input
            bind:value={val}
            placeholder={colType === 'number' ? 'number…' : 'value…'}
            class="h-8 flex-1 text-xs"
            onkeydown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                addFilter();
              }
            }}
          />
          <Button type="button" size="default" disabled={!colName || !val.trim()} onclick={addFilter}>
            <Plus class="size-4" /> Add
          </Button>
        </div>
        <span class="text-[10px] text-muted-foreground/70">
          Pick a column, an operator, a value, then Add — the search re-runs immediately.
        </span>
      </div>

      <!-- Language quick filter -->
      <label class="flex items-center justify-between gap-2">
        <span class="text-muted-foreground">Language</span>
        <select
          bind:value={language}
          onchange={commit}
          class="h-8 w-32 rounded-md border border-border bg-background px-2 text-xs text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <option value="">Any</option>
          <option value="sv">Swedish</option>
          <option value="en">English</option>
        </select>
      </label>

      <!-- Manage which columns appear in the builder -->
      <div class="border-t border-border pt-2">
        <button
          type="button"
          onclick={() => (manageOpen = !manageOpen)}
          class="flex w-full items-center justify-between text-[11px] text-muted-foreground hover:text-foreground"
        >
          <span>Manage columns ({visibleCols.length}/{columns.length} shown)</span>
          <span class="text-[10px]">{manageOpen ? 'hide' : 'show'}</span>
        </button>
        {#if manageOpen}
          <div class="mt-1.5 flex flex-wrap gap-1">
            {#each columns as c (c.name)}
              {@const isHidden = hidden.has(c.name)}
              <button
                type="button"
                onclick={() => toggleHide(c.name)}
                title={isHidden ? `Show '${c.name}'` : `Hide '${c.name}'`}
                class={cn(
                  'inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[11px] transition-colors',
                  isHidden
                    ? 'border-dashed border-border text-muted-foreground/60 hover:text-foreground'
                    : 'border-border text-foreground hover:bg-muted',
                )}
              >
                {#if isHidden}<EyeOff class="size-3" />{:else}<Eye class="size-3" />{/if}
                {c.name}
              </button>
            {/each}
            {#if columns.length === 0}
              <span class="text-[11px] text-muted-foreground">Loading columns…</span>
            {/if}
          </div>
        {/if}
      </div>

      <!-- Advanced raw SQL escape hatch -->
      <details class="border-t border-border pt-2">
        <summary class="cursor-pointer text-[11px] text-muted-foreground hover:text-foreground">
          Advanced — raw SQL (WHERE)
        </summary>
        <textarea
          bind:value={whereSql}
          onchange={commit}
          onblur={commit}
          rows={2}
          placeholder="duration > 60 AND namn LIKE '%alkohol%'"
          class="mt-1.5 min-h-[2rem] w-full resize-y rounded-md border border-border bg-background px-2 py-1.5 font-mono text-[11px] text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        ></textarea>
        <span class="text-[10px] text-muted-foreground/70">
          The builder above writes here. Edit directly for OR / parentheses / functions.
        </span>
      </details>
    </Popover.Content>
  </Popover.Portal>
</Popover.Root>
