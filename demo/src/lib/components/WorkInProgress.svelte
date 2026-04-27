<script lang="ts">
	import Construction from 'lucide-svelte/icons/construction';

	interface Props {
		title: string;
		description: string;
	}

	let { title, description }: Props = $props();
</script>

<main class="flex flex-1 items-center justify-center p-4 md:p-6 lg:p-8">
	<div
		class="bg-card/95 w-full max-w-md space-y-3 rounded-2xl border p-8 text-center shadow-sm backdrop-blur-md"
	>
		<Construction class="text-muted-foreground mx-auto size-10" />
		<h1 class="text-lg font-semibold">{title}</h1>
		<p class="text-muted-foreground text-pretty text-sm">{description}</p>
		<p class="text-muted-foreground pt-2 text-xs">
			Work in progress — use the <span class="text-foreground font-medium">Realtime speech</span>
			tab for a working demo.
		</p>
	</div>
</main>
