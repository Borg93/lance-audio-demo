<script lang="ts">
	import { Progress as ProgressPrimitive } from 'bits-ui';
	import { cn } from '$lib/utils';

	type Props = ProgressPrimitive.RootProps & {
		class?: string;
	};

	let {
		class: className,
		value = 0,
		max = 100,
		ref = $bindable(null),
		...rest
	}: Props = $props();
</script>

<ProgressPrimitive.Root
	bind:ref
	class={cn(
		'bg-secondary relative h-2 w-full overflow-hidden rounded-full',
		className
	)}
	{value}
	{max}
	{...rest}
>
	<div
		class="bg-primary h-full w-full flex-1 transition-all"
		style:transform="translateX(-{100 - (100 * (Number(value) ?? 0)) / (max ?? 1)}%)"
	></div>
</ProgressPrimitive.Root>
