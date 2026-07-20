<script lang="ts">
  // AI-assist prompt bar (floating over the canvas). Type a class → GroundingDINO
  // returns boxes on the page as predictions (status=prediction, source=model:…),
  // reviewed like any other. The interactive-assist round-trip. (ra-atr parity.)
  import { Sparkles } from 'lucide-svelte';
  import { Button, Input } from '$lib/components/ui';
  import type { AnnotatorController } from '../annotator.svelte';

  let { controller }: { controller: AnnotatorController } = $props();

  let prompt = $state('');
  function run(): void {
    if (prompt.trim()) void controller.assist(prompt);
  }
</script>

<div
  class="pointer-events-auto absolute left-1/2 top-2 z-10 flex -translate-x-1/2 items-center gap-1 rounded-lg border border-border bg-card/90 p-1 shadow-md backdrop-blur"
  data-testid="ai-assist"
>
  <Sparkles class="ml-1 size-3.5 text-muted-foreground" />
  <Input
    bind:value={prompt}
    placeholder="AI detect… (e.g. 'text line')"
    class="h-7 w-48"
    onkeydown={(e) => e.key === 'Enter' && run()}
  />
  <Button variant="default" size="sm" disabled={!prompt.trim() || controller.saving} onclick={run}>
    Detect
  </Button>
</div>
