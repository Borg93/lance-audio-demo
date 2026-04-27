<script lang="ts">
	import { Tabs as TabsPrimitive } from 'bits-ui';

	let { ref = $bindable(null), ...rest }: TabsPrimitive.RootProps = $props();
</script>

<TabsPrimitive.Root bind:ref {...rest} />
