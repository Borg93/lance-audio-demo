<script lang="ts">
	import { Tabs as TabsPrimitive } from 'bits-ui';
	import { cn } from '$lib/utils';

	type Props = TabsPrimitive.TriggerProps & { class?: string };
	let { class: className, ref = $bindable(null), ...rest }: Props = $props();
</script>

<TabsPrimitive.Trigger
	bind:ref
	class={cn(
		'ring-offset-background focus-visible:ring-ring data-[state=active]:bg-background data-[state=active]:text-foreground inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1 text-sm font-medium whitespace-nowrap transition-all focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 data-[state=active]:shadow',
		className
	)}
	{...rest}
/>
