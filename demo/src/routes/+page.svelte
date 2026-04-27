<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import Mic from 'lucide-svelte/icons/mic';
	import Loader2 from 'lucide-svelte/icons/loader-2';
	import CircleX from 'lucide-svelte/icons/circle-x';
	import Cpu from 'lucide-svelte/icons/cpu';
	import Zap from 'lucide-svelte/icons/zap';
	import Layers from 'lucide-svelte/icons/layers';

	import ProgressItem from '$lib/components/ProgressItem.svelte';
	import TranscriptHistory from '$lib/components/TranscriptHistory.svelte';
	import ModelPicker from '$lib/components/ModelPicker.svelte';
	import RealtimePanel from '$lib/components/RealtimePanel.svelte';
	import BatchPanel from '$lib/components/BatchPanel.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tabs from '$lib/components/ui/tabs';

	import { type ComputeDevice, type BackendPreference } from '$lib/webgpu';
	import {
		LANGUAGES,
		DEFAULT_LANGUAGE,
		findVariant,
		type ModelKey,
		type SizeKey
	} from '$lib/models';
	import { type TranscriptEntry } from '$lib/transcript';

	type ProgressEvent = { file: string; progress?: number; total?: number; loaded?: number };

	let webgpuAvailable = $state(false);

	let selectedLanguage = $state<ModelKey>(DEFAULT_LANGUAGE);
	let selectedSize = $state<SizeKey>(LANGUAGES[DEFAULT_LANGUAGE].defaultSize);
	let backendPref = $state<BackendPreference>('auto');
	const languagePack = $derived(LANGUAGES[selectedLanguage]);
	const modelSpec = $derived(findVariant(selectedLanguage, selectedSize));

	let status = $state<'loading' | 'ready' | 'error' | null>(null);
	let loadingMessage = $state('');
	let errorMessage = $state('');
	let progressItems = $state<ProgressEvent[]>([]);
	let device = $state<ComputeDevice | null>(null);

	let history = $state<TranscriptEntry[]>([]);
	let activeTab = $state<'realtime' | 'batch'>('realtime');

	let worker = $state<Worker | null>(null);

	const workerReady = $derived(status === 'ready' && worker !== null);

	function spawnWorker() {
		worker?.terminate();
		worker = new Worker(new URL('$lib/worker.ts', import.meta.url), { type: 'module' });

		worker.addEventListener('message', (e: MessageEvent) => {
			const d = e.data;
			// Handle model lifecycle messages only. Inference messages are
			// routed to RealtimePanel / BatchPanel via their own listeners.
			switch (d.status) {
				case 'loading':
					status = 'loading';
					loadingMessage = d.data;
					break;
				case 'device':
					device = d.device;
					break;
				case 'initiate':
					if (!progressItems.some((item) => item.file === d.file)) {
						progressItems = [...progressItems, d];
					}
					break;
				case 'progress':
					progressItems = progressItems.map((item) =>
						item.file === d.file ? { ...item, ...d } : item
					);
					break;
				case 'done':
					progressItems = progressItems.filter((item) => item.file !== d.file);
					break;
				case 'ready':
					status = 'ready';
					break;
				case 'error':
					// Load/device errors are global; inference errors always include a
					// jobId (batch) or have been handled by the panel already.
					if (d.jobId === undefined && !workerReady) {
						status = 'error';
						errorMessage = d.data;
					}
					break;
			}
		});
	}

	onMount(() => {
		webgpuAvailable = !!(navigator as Navigator & { gpu?: unknown }).gpu;
		if (webgpuAvailable) spawnWorker();
	});

	onDestroy(() => {
		worker?.terminate();
		worker = null;
	});

	function loadModel() {
		status = 'loading';
		progressItems = [];
		worker?.postMessage({
			type: 'load',
			data: {
				spec: {
					modelId: modelSpec.modelId,
					language: languagePack.language,
					melBins: modelSpec.melBins,
					dtype: modelSpec.dtype
				},
				backendPref
			}
		});
	}

	function resetForModelSwap() {
		if (status === null) return;
		progressItems = [];
		device = null;
		status = null;
		errorMessage = '';
		spawnWorker();
	}

	function onLanguageChange(next: ModelKey) {
		if (next === selectedLanguage) return;
		selectedLanguage = next;
		selectedSize = LANGUAGES[next].defaultSize;
		resetForModelSwap();
	}

	function onSizeChange(next: SizeKey) {
		if (next === selectedSize) return;
		selectedSize = next;
		resetForModelSwap();
	}

	function onBackendChange(next: BackendPreference) {
		if (next === backendPref) return;
		backendPref = next;
		resetForModelSwap();
	}

	function retry() {
		status = null;
		errorMessage = '';
		progressItems = [];
	}

	function addEntry(entry: TranscriptEntry) {
		history = [entry, ...history];
	}

	function deleteHistoryEntry(id: string) {
		history = history.filter((e) => e.id !== id);
	}
</script>

{#if webgpuAvailable}
	<main class="flex flex-1 justify-center p-4 md:p-6 lg:p-8">
		<article
			class="bg-card/95 relative w-full max-w-5xl overflow-hidden rounded-2xl border shadow-sm backdrop-blur-md"
		>
			<!-- Header -->
			<header class="border-b px-6 py-6 md:px-10">
				<div
					class="flex w-full flex-col gap-4 md:flex-row md:items-start md:justify-between"
				>
					<div class="flex-1 space-y-1.5">
						<h1 class="text-2xl font-semibold tracking-tight md:text-3xl">
							Speech-to-text
						</h1>
						<p class="text-muted-foreground max-w-2xl text-sm">
							Everything runs on <span class="text-foreground font-medium">your own computer</span>
							— your CPU, your GPU, your memory. Audio, model weights and transcription never leave
							this device, so there is nothing to upload, nothing to intercept, and nothing for
							anyone else to see.
						</p>
					</div>
					<div class="flex flex-wrap items-center gap-2 text-[11px]">
						<span
							class="inline-flex items-center gap-1.5 rounded-md border bg-card/50 px-2 py-1"
						>
							{#if status === 'ready'}
								<span class="bg-emerald-500 inline-block size-1.5 rounded-full"></span>
								Model loaded
							{:else if status === 'loading'}
								<Loader2 class="size-3 animate-spin" />
								Loading…
							{:else if status === 'error'}
								<span class="bg-destructive inline-block size-1.5 rounded-full"></span>
								Error
							{:else}
								<span class="bg-muted-foreground/40 inline-block size-1.5 rounded-full"></span>
								Idle
							{/if}
						</span>
						{#if device}
							<span
								class="inline-flex items-center gap-1 rounded-md border bg-card/50 px-2 py-1 uppercase"
								title="Compute backend"
							>
								{#if device === 'webgpu'}
									<Zap class="size-3" /> WebGPU
								{:else}
									<Cpu class="size-3" /> WASM
								{/if}
							</span>
						{/if}
						<span
							class="text-muted-foreground inline-flex items-center gap-1 rounded-md border bg-card/50 px-2 py-1 font-mono"
							title="Active model"
						>
							{modelSpec.modelId}
						</span>
					</div>
				</div>
			</header>

			<!-- Content -->
			<div class="flex w-full flex-col gap-8 px-6 py-8 md:px-10">
				<ModelPicker
					{selectedLanguage}
					{selectedSize}
					{backendPref}
					disabled={status === 'loading'}
					{onLanguageChange}
					{onSizeChange}
					{onBackendChange}
				/>

				<!-- Pre-load / loading / error states -->
				{#if status === null}
					<section
						class="bg-muted/40 flex flex-col gap-4 rounded-xl border p-6 md:flex-row md:items-center md:justify-between"
					>
						<div class="space-y-1 text-sm">
							<p class="text-foreground font-medium">{modelSpec.modelId}</p>
							<p class="text-muted-foreground max-w-xl">{languagePack.description}</p>
							<p class="text-muted-foreground text-xs">
								<span class="text-foreground font-medium"
									>{modelSpec.sizeLabel} · {modelSpec.sizeNote}</span
								> first download, cached in the browser.
							</p>
						</div>
						<Button size="lg" onclick={loadModel} class="shrink-0">
							<Mic />
							Load {languagePack.label} model
						</Button>
					</section>
				{/if}

				{#if status === 'loading'}
					<section class="bg-muted/40 space-y-4 rounded-xl border p-6">
						<div class="flex items-center gap-2 text-sm">
							<Loader2 class="animate-spin" />
							<span>{loadingMessage}</span>
						</div>
						{#if progressItems.length > 0}
							<div class="space-y-1">
								{#each progressItems as item (item.file)}
									<ProgressItem
										text={item.file}
										percentage={item.progress}
										total={item.total}
									/>
								{/each}
							</div>
						{/if}
					</section>
				{/if}

				{#if status === 'error'}
					<section
						class="border-destructive/40 bg-destructive/10 text-destructive flex items-start gap-2 rounded-xl border p-4 text-sm"
					>
						<CircleX class="mt-0.5 size-4 shrink-0" />
						<div class="flex-1 space-y-2">
							<p class="font-medium">Something went wrong</p>
							<p class="text-destructive/80 break-words">{errorMessage}</p>
							<Button variant="outline" size="sm" onclick={retry}>Try again</Button>
						</div>
					</section>
				{/if}

				<!-- Mode tabs (Realtime / Batch) -->
				<Tabs.Root
					value={activeTab}
					onValueChange={(v) => v && (activeTab = v as 'realtime' | 'batch')}
				>
					<Tabs.List class="mb-6">
						<Tabs.Trigger value="realtime">
							<Mic class="size-4" /> Realtime
						</Tabs.Trigger>
						<Tabs.Trigger value="batch">
							<Layers class="size-4" /> Batch
						</Tabs.Trigger>
					</Tabs.List>

					<Tabs.Content value="realtime">
						<RealtimePanel
							{worker}
							{workerReady}
							{modelSpec}
							{languagePack}
							onEntry={addEntry}
						/>
					</Tabs.Content>

					<Tabs.Content value="batch">
						<BatchPanel
							{worker}
							{workerReady}
							{modelSpec}
							{languagePack}
							onEntry={addEntry}
						/>
					</Tabs.Content>
				</Tabs.Root>

				<TranscriptHistory entries={history} onDelete={deleteHistoryEntry} />
			</div>
		</article>
	</main>
{:else}
	<main class="flex flex-1 items-center justify-center p-6">
		<div
			class="bg-card/95 backdrop-blur-md w-full max-w-sm space-y-2 rounded-xl border p-6 text-center shadow-sm"
		>
			<CircleX class="text-destructive mx-auto size-10" />
			<h1 class="text-lg font-semibold">WebGPU not available</h1>
			<p class="text-muted-foreground text-sm">
				This demo requires a WebGPU-capable browser (Chrome/Edge 113+, or Safari Technology
				Preview).
			</p>
		</div>
	</main>
{/if}
