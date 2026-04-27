<script lang="ts">
	import { Select as SelectPrimitive, type WithoutChild } from 'bits-ui';
	import Check from 'lucide-svelte/icons/check';
	import { cn } from '$lib/utils';

	type Props = WithoutChild<SelectPrimitive.ItemProps> & {
		class?: string;
	};

	let {
		class: className,
		value,
		label,
		disabled,
		ref = $bindable(null),
		children: childrenProp,
		...rest
	}: Props = $props();
</script>

<SelectPrimitive.Item
	bind:ref
	{value}
	{label}
	{disabled}
	class={cn(
		'focus:bg-accent focus:text-accent-foreground relative flex w-full cursor-default items-center rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
		className
	)}
	{...rest}
>
	{#snippet children({ selected, highlighted })}
		<span class="absolute right-2 flex size-3.5 items-center justify-center">
			{#if selected}
				<Check class="size-4" />
			{/if}
		</span>
		{#if childrenProp}
			{@render childrenProp({ selected, highlighted })}
		{:else}
			{label ?? value}
		{/if}
	{/snippet}
</SelectPrimitive.Item>
