<script lang="ts">
	import { Select as SelectPrimitive } from 'bits-ui';

	let { children, ...rest }: SelectPrimitive.RootProps = $props();
</script>

<SelectPrimitive.Root {...rest}>
	{@render children?.()}
</SelectPrimitive.Root>
