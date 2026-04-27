<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { cn } from '$lib/utils';

	type Props = HTMLAttributes<HTMLDivElement> & { class?: string };
	let { class: className, children, ...rest }: Props = $props();
</script>

<div
	class={cn(
		'bg-card text-card-foreground rounded-xl border shadow-sm',
		className
	)}
	{...rest}
>
	{@render children?.()}
</div>
