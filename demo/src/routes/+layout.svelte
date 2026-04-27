<script lang="ts">
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import Sidebar from '$lib/components/Sidebar.svelte';
	import { DotPattern } from '$lib/components/magic';

	let { children } = $props();
</script>

<ModeWatcher defaultMode="system" />

<div class="bg-background text-foreground flex min-h-screen">
	<Sidebar />
	<div class="relative flex min-h-screen flex-1 flex-col overflow-hidden">
		<!-- Decorative dot pattern lives behind the main content only, not
		     behind the sidebar. -->
		<DotPattern
			width={22}
			height={22}
			cr={1.2}
			class="[mask-image:radial-gradient(ellipse_at_50%_30%,black_40%,transparent_90%)] fill-neutral-400/50 dark:fill-neutral-500/40"
		/>
		<div class="relative z-10 flex min-h-screen flex-1 flex-col">
			{@render children()}
		</div>
	</div>
</div>
