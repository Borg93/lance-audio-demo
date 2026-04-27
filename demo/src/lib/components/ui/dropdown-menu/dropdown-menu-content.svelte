<script lang="ts">
	import { DropdownMenu as DropdownMenuPrimitive, type WithoutChild } from 'bits-ui';
	import { cn } from '$lib/utils';

	type Props = WithoutChild<DropdownMenuPrimitive.ContentProps> & {
		class?: string;
	};

	let {
		class: className,
		sideOffset = 4,
		ref = $bindable(null),
		children,
		...rest
	}: Props = $props();
</script>

<DropdownMenuPrimitive.Portal>
	<DropdownMenuPrimitive.Content
		bind:ref
		{sideOffset}
		class={cn(
			'bg-popover text-popover-foreground data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 z-50 min-w-[8rem] origin-[--bits-dropdown-menu-content-transform-origin] overflow-hidden rounded-md border p-1 shadow-md outline-hidden',
			className
		)}
		{...rest}
	>
		{@render children?.()}
	</DropdownMenuPrimitive.Content>
</DropdownMenuPrimitive.Portal>
