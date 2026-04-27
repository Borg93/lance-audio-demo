<script lang="ts">
	import { DropdownMenu as DropdownMenuPrimitive } from 'bits-ui';
	import { cn } from '$lib/utils';

	type Props = DropdownMenuPrimitive.ItemProps & {
		class?: string;
		inset?: boolean;
	};

	let {
		class: className,
		inset,
		ref = $bindable(null),
		children,
		...rest
	}: Props = $props();
</script>

<DropdownMenuPrimitive.Item
	bind:ref
	class={cn(
		'focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 relative flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
		inset && 'pl-8',
		className
	)}
	{...rest}
>
	{@render children?.()}
</DropdownMenuPrimitive.Item>
