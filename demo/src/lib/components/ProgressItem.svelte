<script lang="ts">
	import { Progress } from '$lib/components/ui/progress';

	interface Props {
		text: string;
		percentage?: number;
		total?: number;
	}

	let { text, percentage = 0, total }: Props = $props();

	function formatBytes(size: number): string {
		const i = size === 0 ? 0 : Math.floor(Math.log(size) / Math.log(1024));
		return (
			+(size / Math.pow(1024, i)).toFixed(2) * 1 + ['B', 'kB', 'MB', 'GB', 'TB'][i]
		);
	}
</script>

<div class="mb-2 space-y-1">
	<div class="text-muted-foreground flex justify-between text-xs">
		<span class="truncate pr-2">{text}</span>
		<span class="tabular-nums">
			{percentage.toFixed(1)}%{total != null && !isNaN(total) ? ` · ${formatBytes(total)}` : ''}
		</span>
	</div>
	<Progress value={percentage} />
</div>
