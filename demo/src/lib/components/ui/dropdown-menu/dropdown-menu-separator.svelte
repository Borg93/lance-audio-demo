<script lang="ts">
	import { DropdownMenu as DropdownMenuPrimitive } from 'bits-ui';
	import { cn } from '$lib/utils';

	type Props = DropdownMenuPrimitive.SeparatorProps & { class?: string };

	let { class: className, ref = $bindable(null), ...rest }: Props = $props();
</script>

<DropdownMenuPrimitive.Separator
	bind:ref
	class={cn('bg-muted -mx-1 my-1 h-px', className)}
	{...rest}
/>
