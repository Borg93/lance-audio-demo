<script lang="ts">
	import { DropdownMenu as DropdownMenuPrimitive, type WithoutChild } from 'bits-ui';
	import Circle from 'lucide-svelte/icons/circle';
	import { cn } from '$lib/utils';

	type Props = WithoutChild<DropdownMenuPrimitive.RadioItemProps> & {
		class?: string;
	};

	let {
		class: className,
		value,
		ref = $bindable(null),
		children: childrenProp,
		...rest
	}: Props = $props();
</script>

<DropdownMenuPrimitive.RadioItem
	bind:ref
	{value}
	class={cn(
		'focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 relative flex cursor-default items-center gap-2 rounded-sm py-1.5 pr-2 pl-8 text-sm outline-hidden select-none [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
		className
	)}
	{...rest}
>
	{#snippet children({ checked })}
		<span class="absolute left-2 flex size-3.5 items-center justify-center">
			{#if checked}
				<Circle class="size-2 fill-current" />
			{/if}
		</span>
		{@render childrenProp?.({ checked })}
	{/snippet}
</DropdownMenuPrimitive.RadioItem>
