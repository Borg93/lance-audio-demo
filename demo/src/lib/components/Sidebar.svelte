<script lang="ts">
	import { page } from '$app/state';
	import Mic from 'lucide-svelte/icons/mic';
	import Search from 'lucide-svelte/icons/search';
	import Waves from 'lucide-svelte/icons/audio-waveform';
	import Info from 'lucide-svelte/icons/info';
	import PanelLeftClose from 'lucide-svelte/icons/panel-left-close';
	import PanelLeftOpen from 'lucide-svelte/icons/panel-left-open';
	import ThemeToggle from './ThemeToggle.svelte';
	import { cn } from '$lib/utils';

	const items = [
		{ href: '/', label: 'Speech-to-text', icon: Mic, wip: false },
		{ href: '/search', label: 'Multimodal search', icon: Search, wip: true }
	];

	const footerItems = [{ href: '/about', label: 'About', icon: Info, wip: false }];

	let collapsed = $state(false);

	// Load preference once (SSR-safe — this file only runs in the browser
	// because the parent layout sets ssr=false on the only non-WIP route).
	$effect(() => {
		const saved = typeof localStorage !== 'undefined' ? localStorage.getItem('sidebar-collapsed') : null;
		if (saved === '1') collapsed = true;
	});

	function toggle() {
		collapsed = !collapsed;
		try {
			localStorage.setItem('sidebar-collapsed', collapsed ? '1' : '0');
		} catch {
			// localStorage can throw in incognito / quota-exceeded — ignore
		}
	}
</script>

<aside
	class={cn(
		'bg-muted/30 hidden shrink-0 flex-col border-r transition-[width] duration-200 md:flex',
		collapsed ? 'w-14' : 'w-56'
	)}
	aria-label="Primary navigation"
>
	<div class={cn('flex items-center border-b px-2 py-3', collapsed ? 'justify-center' : 'justify-between px-4')}>
		{#if !collapsed}
			<a href="/" class="flex items-center gap-2">
				<Waves class="size-5" />
				<span class="flex flex-col leading-none">
					<span class="text-sm font-semibold tracking-tight">Multimodal lab</span>
					<span class="text-muted-foreground text-[9px] tracking-[0.15em] uppercase">
						local · webgpu
					</span>
				</span>
			</a>
		{/if}
		<button
			type="button"
			onclick={toggle}
			title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
			class="text-muted-foreground hover:bg-background/60 hover:text-foreground flex size-8 items-center justify-center rounded-md transition-colors"
		>
			{#if collapsed}
				<PanelLeftOpen class="size-4" />
			{:else}
				<PanelLeftClose class="size-4" />
			{/if}
		</button>
	</div>

	<nav class={cn('flex flex-col gap-1', collapsed ? 'p-1.5' : 'p-2')}>
		{#each items as item (item.href)}
			{@const Icon = item.icon}
			{@const active = page.url.pathname === item.href}
			<a
				href={item.href}
				aria-current={active ? 'page' : undefined}
				title={collapsed ? item.label : undefined}
				class={cn(
					'group flex items-center rounded-md text-sm transition-colors',
					collapsed ? 'size-8 justify-center' : 'justify-between px-3 py-2',
					active
						? 'bg-background text-foreground border shadow-sm'
						: 'text-muted-foreground hover:bg-background/60 hover:text-foreground'
				)}
			>
				<span class={cn('flex items-center', collapsed ? '' : 'gap-2.5')}>
					<Icon class="size-4" />
					{#if !collapsed}
						<span>{item.label}</span>
					{/if}
				</span>
				{#if item.wip && !collapsed}
					<span
						class="bg-muted text-muted-foreground rounded px-1.5 py-0.5 text-[9px] font-medium tracking-wider uppercase"
					>
						WIP
					</span>
				{/if}
			</a>
		{/each}
	</nav>

	<div
		class={cn(
			'mt-auto flex flex-col border-t',
			collapsed ? 'items-center gap-3 p-2' : 'gap-2 p-2'
		)}
	>
		<nav class={cn('flex flex-col', collapsed ? 'items-center gap-1' : 'gap-1')}>
			{#each footerItems as item (item.href)}
				{@const Icon = item.icon}
				{@const active = page.url.pathname === item.href}
				<a
					href={item.href}
					aria-current={active ? 'page' : undefined}
					title={collapsed ? item.label : undefined}
					class={cn(
						'group flex items-center rounded-md text-sm transition-colors',
						collapsed ? 'size-8 justify-center' : 'gap-2.5 px-3 py-2',
						active
							? 'bg-background text-foreground border shadow-sm'
							: 'text-muted-foreground hover:bg-background/60 hover:text-foreground'
					)}
				>
					<Icon class="size-4" />
					{#if !collapsed}
						<span>{item.label}</span>
					{/if}
				</a>
			{/each}
		</nav>
		<div class={cn(collapsed ? '' : 'px-1 pb-1')}>
			<ThemeToggle {collapsed} />
		</div>
	</div>
</aside>
