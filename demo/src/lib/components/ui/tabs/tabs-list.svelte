<script lang="ts">
	import { Tabs as TabsPrimitive } from 'bits-ui';
	import { cn } from '$lib/utils';

	type Props = TabsPrimitive.ListProps & { class?: string };
	let { class: className, ref = $bindable(null), ...rest }: Props = $props();
</script>

<TabsPrimitive.List
	bind:ref
	class={cn(
		'bg-muted text-muted-foreground inline-flex h-9 items-center justify-center rounded-lg p-1',
		className
	)}
	{...rest}
/>
