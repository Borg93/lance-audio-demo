<script lang="ts">
	import { Select as SelectPrimitive } from 'bits-ui';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import { cn } from '$lib/utils';

	type Props = SelectPrimitive.TriggerProps & { class?: string };

	let {
		class: className,
		ref = $bindable(null),
		children,
		...rest
	}: Props = $props();
</script>

<SelectPrimitive.Trigger
	bind:ref
	class={cn(
		'border-input bg-background ring-offset-background placeholder:text-muted-foreground focus-visible:ring-ring data-[placeholder]:text-muted-foreground flex h-9 w-full items-center justify-between rounded-md border px-3 py-2 text-sm whitespace-nowrap shadow-sm transition-colors focus-visible:ring-1 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1 [&[data-state=open]>svg]:rotate-180',
		className
	)}
	{...rest}
>
	{@render children?.()}
	<ChevronDown class="size-4 shrink-0 opacity-50 transition-transform duration-200" />
</SelectPrimitive.Trigger>
