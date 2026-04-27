<script lang="ts">
  import '../app.css';
  import ThemeToggle from '$lib/components/theme-toggle.svelte';
  import StatusBadge from '$lib/components/status-badge.svelte';
  import type { Snippet } from 'svelte';

  let { children }: { children: Snippet } = $props();
</script>

<div class="grid h-screen grid-rows-[auto_1fr] bg-background text-foreground">
  <!--
    Tiny header — single row, ~40px tall. Brand on the left, theme on the
    right. No nav (single-page app), no padding waste.
  -->
  <header class="flex h-10 items-center justify-between border-b border-border bg-card/60 px-4">
    <a
      href="/"
      class="text-xs font-semibold tracking-tight text-foreground/90 hover:text-foreground"
    >
      raudio<span class="text-muted-foreground font-normal"> · viewer</span>
    </a>
    <div class="flex items-center gap-1">
      <StatusBadge />
      <ThemeToggle />
    </div>
  </header>

  <main class="min-h-0 overflow-hidden">
    {@render children()}
  </main>
</div>
