<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { cn } from '$lib/utils';

	type Props = HTMLAttributes<HTMLDivElement> & {
		class?: string;
		inset?: boolean;
	};

	let { class: className, inset, children, ...rest }: Props = $props();
</script>

<div class={cn('px-2 py-1.5 text-sm font-semibold', inset && 'pl-8', className)} {...rest}>
	{@render children?.()}
</div>
