<script lang="ts">
	import { cn } from '$lib/utils';

	interface Props {
		width?: number;
		height?: number;
		x?: number;
		y?: number;
		cx?: number;
		cy?: number;
		cr?: number;
		glow?: boolean;
		class?: string;
	}

	let {
		width = 16,
		height = 16,
		x = 0,
		y = 0,
		cx = 1,
		cy = 1,
		cr = 1,
		glow = false,
		class: className = ''
	}: Props = $props();

	const id = `dot-pattern-${Math.random().toString(36).slice(2, 9)}`;
</script>

<svg
	aria-hidden="true"
	class={cn(
		'pointer-events-none absolute inset-0 h-full w-full',
		'fill-neutral-400/40 dark:fill-neutral-500/30',
		glow && 'animate-pulse',
		className
	)}
>
	<defs>
		<pattern {id} {width} {height} {x} {y} patternUnits="userSpaceOnUse">
			<circle {cx} {cy} r={cr} />
		</pattern>
	</defs>
	<rect width="100%" height="100%" fill="url(#{id})" />
</svg>
