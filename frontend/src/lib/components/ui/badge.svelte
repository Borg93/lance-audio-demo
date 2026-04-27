<script lang="ts">
  import { cn } from '$lib/utils';
  import type { Snippet } from 'svelte';
  import type { HTMLAttributes } from 'svelte/elements';

  type Props = HTMLAttributes<HTMLSpanElement> & { class?: string; children: Snippet };
  let { class: className, children, ...rest }: Props = $props();
</script>

<span
  class={cn(
    'inline-flex items-center rounded-full border border-border px-2 py-0.5 text-xs font-medium text-muted-foreground',
    className,
  )}
  {...rest}
>
  {@render children()}
</span>
