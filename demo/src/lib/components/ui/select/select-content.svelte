<script lang="ts">
	import { Select as SelectPrimitive, type WithoutChild } from 'bits-ui';
	import { cn } from '$lib/utils';

	type Props = WithoutChild<SelectPrimitive.ContentProps> & {
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

<SelectPrimitive.Portal>
	<SelectPrimitive.Content
		bind:ref
		{sideOffset}
		class={cn(
			'bg-popover text-popover-foreground data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 relative z-50 max-h-[--bits-select-content-available-height] min-w-[8rem] origin-[--bits-select-content-transform-origin] overflow-hidden rounded-md border shadow-md outline-hidden',
			className
		)}
		{...rest}
	>
		<SelectPrimitive.Viewport class="p-1">
			{@render children?.()}
		</SelectPrimitive.Viewport>
	</SelectPrimitive.Content>
</SelectPrimitive.Portal>
