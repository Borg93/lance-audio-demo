<script lang="ts">
  import { Checkbox as CheckboxPrimitive, type WithoutChildrenOrChild } from 'bits-ui';
  import { Check, Minus } from 'lucide-svelte';
  import { cn } from '$lib/utils';

  type Props = WithoutChildrenOrChild<CheckboxPrimitive.RootProps> & { class?: string };
  let {
    class: className,
    checked = $bindable(false),
    indeterminate = $bindable(false),
    ref = $bindable(null),
    ...rest
  }: Props = $props();
</script>

<CheckboxPrimitive.Root
  bind:ref
  bind:checked
  bind:indeterminate
  class={cn(
    'peer size-4 shrink-0 rounded-sm border border-border bg-input',
    'data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
    'disabled:cursor-not-allowed disabled:opacity-50',
    className,
  )}
  {...rest}
>
  {#snippet children({ checked, indeterminate })}
    <div class="flex items-center justify-center text-current">
      {#if indeterminate}<Minus class="size-3" />{:else if checked}<Check class="size-3" />{/if}
    </div>
  {/snippet}
</CheckboxPrimitive.Root>
