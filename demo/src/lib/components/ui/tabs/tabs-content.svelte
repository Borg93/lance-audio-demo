<script lang="ts">
	import { Tabs as TabsPrimitive } from 'bits-ui';
	import { cn } from '$lib/utils';

	type Props = TabsPrimitive.ContentProps & { class?: string };
	let { class: className, ref = $bindable(null), ...rest }: Props = $props();
</script>

<TabsPrimitive.Content
	bind:ref
	class={cn(
		'ring-offset-background focus-visible:ring-ring focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-hidden',
		className
	)}
	{...rest}
/>
