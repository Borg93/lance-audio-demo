<script lang="ts">
	import { mode, setMode } from 'mode-watcher';
	import Sun from 'lucide-svelte/icons/sun';
	import Moon from 'lucide-svelte/icons/moon';
	import Monitor from 'lucide-svelte/icons/monitor';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { cn } from '$lib/utils';

	interface Props {
		collapsed?: boolean;
	}

	let { collapsed = false }: Props = $props();

	const options = [
		{ value: 'light', icon: Sun, label: 'Light' },
		{ value: 'dark', icon: Moon, label: 'Dark' },
		{ value: 'system', icon: Monitor, label: 'System' }
	] as const;

	const current = $derived(options.find((o) => o.value === mode.current) ?? options[2]);

	function handleChange(next: string) {
		setMode(next as 'light' | 'dark' | 'system');
	}
</script>

<DropdownMenu.Root>
	<DropdownMenu.Trigger
		class={cn(
			'hover:bg-background/60 hover:text-foreground text-muted-foreground inline-flex items-center gap-2 rounded-md border px-2 py-1.5 text-xs transition-colors',
			collapsed ? 'w-10 justify-center px-0' : 'w-full justify-between'
		)}
		aria-label="Theme"
	>
		{#if collapsed}
			<current.icon class="size-4" />
		{:else}
			<span class="inline-flex items-center gap-2">
				<current.icon class="size-3.5" />
				<span>Theme</span>
			</span>
			<span class="text-foreground text-[10px] font-medium tracking-wider uppercase">
				{current.label}
			</span>
		{/if}
	</DropdownMenu.Trigger>

	<DropdownMenu.Content align={collapsed ? 'start' : 'end'} sideOffset={6} class="w-40">
		<DropdownMenu.Label>Appearance</DropdownMenu.Label>
		<DropdownMenu.Separator />
		<DropdownMenu.RadioGroup value={mode.current ?? 'system'} onValueChange={handleChange}>
			{#each options as opt (opt.value)}
				<DropdownMenu.RadioItem value={opt.value}>
					<opt.icon class="size-3.5 opacity-70" />
					<span>{opt.label}</span>
				</DropdownMenu.RadioItem>
			{/each}
		</DropdownMenu.RadioGroup>
	</DropdownMenu.Content>
</DropdownMenu.Root>
