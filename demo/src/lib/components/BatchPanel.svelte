<script lang="ts">
	import Loader2 from 'lucide-svelte/icons/loader-2';
	import Upload from 'lucide-svelte/icons/upload';
	import FileAudio from 'lucide-svelte/icons/file-audio';
	import FileVideo from 'lucide-svelte/icons/file-video';
	import CheckCircle2 from 'lucide-svelte/icons/check-circle-2';
	import X from 'lucide-svelte/icons/x';
	import Play from 'lucide-svelte/icons/play';

	import { Button } from './ui/button';
	import { Progress } from './ui/progress';
	import { decodeMediaFile, isMediaFile } from '$lib/audio-decoder';
	import { formatTimestamp, type TranscriptEntry, type TranscriptSegment } from '$lib/transcript';
	import type { LanguagePack, ModelVariant } from '$lib/models';

	interface Props {
		worker: Worker | null;
		workerReady: boolean;
		modelSpec: ModelVariant;
		languagePack: LanguagePack;
		onEntry: (entry: TranscriptEntry) => void;
	}

	let { worker, workerReady, modelSpec, languagePack, onEntry }: Props = $props();

	type Job = {
		id: string;
		file: File;
		status: 'queued' | 'decoding' | 'transcribing' | 'done' | 'error';
		chunkIndex?: number;
		totalChunks?: number;
		error?: string;
		entryId?: string;
	};

	let jobs = $state<Job[]>([]);
	let runningJobId: string | null = null;
	let batchRunning = $state(false);
	let isDragging = $state(false);
	let fileInput = $state<HTMLInputElement | null>(null);

	const queuedCount = $derived(jobs.filter((j) => j.status === 'queued').length);
	const doneCount = $derived(jobs.filter((j) => j.status === 'done').length);
	const errorCount = $derived(jobs.filter((j) => j.status === 'error').length);
	const processedCount = $derived(doneCount + errorCount);
	const canRun = $derived(workerReady && !batchRunning && queuedCount > 0);

	function jobProgress(job: Job): number {
		if (job.status === 'done') return 100;
		if (job.status === 'error') return 0;
		if (job.status === 'queued') return 0;
		if (job.status === 'decoding') return 2; // show a sliver so the bar appears
		// transcribing:
		const total = job.totalChunks ?? 1;
		const idx = (job.chunkIndex ?? 0) + 1; // current chunk being worked on (1-based)
		return Math.min(100, (idx / total) * 100);
	}

	// Attach listener for job-tagged messages only.
	$effect(() => {
		if (!worker) return;
		const handler = (e: MessageEvent) => {
			const d = e.data;
			if (d.jobId === undefined) return; // not ours
			switch (d.status) {
				case 'finalize-progress':
					updateJob(d.jobId, {
						chunkIndex: d.chunkIndex,
						totalChunks: d.totalChunks
					});
					break;
				case 'finalized':
					handleFinalized(d.jobId, d.text as string, d.segments as TranscriptSegment[]);
					break;
				case 'error':
					updateJob(d.jobId, { status: 'error', error: d.data });
					runNextJob();
					break;
			}
		};
		worker.addEventListener('message', handler);
		return () => worker?.removeEventListener('message', handler);
	});

	function updateJob(id: string, patch: Partial<Job>) {
		jobs = jobs.map((j) => (j.id === id ? { ...j, ...patch } : j));
	}

	function addFiles(files: FileList | File[]) {
		const list = Array.from(files).filter(isMediaFile);
		if (list.length === 0) return;
		const newJobs: Job[] = list.map((f) => ({
			id: crypto.randomUUID(),
			file: f,
			status: 'queued'
		}));
		jobs = [...jobs, ...newJobs];
	}

	function removeJob(id: string) {
		if (runningJobId === id) return;
		jobs = jobs.filter((j) => j.id !== id);
	}

	function clearQueued() {
		jobs = jobs.filter((j) => j.status !== 'queued');
	}

	async function runBatch() {
		if (!canRun) return;
		batchRunning = true;
		await runNextJob();
	}

	async function runNextJob() {
		const next = jobs.find((j) => j.status === 'queued');
		if (!next || !worker) {
			batchRunning = false;
			runningJobId = null;
			return;
		}
		runningJobId = next.id;
		updateJob(next.id, { status: 'decoding' });

		try {
			const audio = await decodeMediaFile(next.file);
			updateJob(next.id, { status: 'transcribing' });
			worker.postMessage({ type: 'finalize', data: { audio, jobId: next.id } }, [audio.buffer]);
		} catch (err) {
			updateJob(next.id, {
				status: 'error',
				error: err instanceof Error ? err.message : String(err)
			});
			await runNextJob();
		}
	}

	function handleFinalized(jobId: string, text: string, segments: TranscriptSegment[]) {
		const job = jobs.find((j) => j.id === jobId);
		if (!job) return;

		const entryId = crypto.randomUUID();
		onEntry({
			id: entryId,
			startedAt: job.file.lastModified || Date.now(),
			durationMs:
				segments.length > 0 ? Math.round(segments[segments.length - 1].end * 1000) : 0,
			modelId: modelSpec.modelId,
			language: languagePack.language,
			audioSource: 'mic',
			text: text.trim(),
			segments,
			audioBlob: job.file,
			audioType: job.file.type || 'audio/webm'
		});

		updateJob(jobId, { status: 'done', entryId });
		runNextJob();
	}

	function onDrop(e: DragEvent) {
		e.preventDefault();
		isDragging = false;
		if (e.dataTransfer?.files.length) addFiles(e.dataTransfer.files);
	}
	function onDragOver(e: DragEvent) {
		e.preventDefault();
		isDragging = true;
	}
	function onDragLeave() {
		isDragging = false;
	}
	function openPicker() {
		fileInput?.click();
	}
	function onFileInputChange(e: Event) {
		const input = e.currentTarget as HTMLInputElement;
		if (input.files) addFiles(input.files);
		input.value = '';
	}

	function formatBytes(n: number): string {
		if (n === 0) return '0 B';
		const i = Math.floor(Math.log(n) / Math.log(1024));
		const units = ['B', 'kB', 'MB', 'GB'];
		return (n / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1) + ' ' + units[i];
	}
</script>

<div class="flex flex-col gap-4">
	<div
		role="button"
		tabindex="0"
		onclick={openPicker}
		onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && openPicker()}
		ondrop={onDrop}
		ondragover={onDragOver}
		ondragleave={onDragLeave}
		class="hover:bg-muted/40 flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-8 text-center transition-colors {isDragging
			? 'border-primary bg-primary/5'
			: 'border-muted-foreground/30'}"
	>
		<Upload class="text-muted-foreground size-6" />
		<p class="text-sm font-medium">
			Drop audio or video files here, or <span class="underline">browse</span>
		</p>
		<p class="text-muted-foreground text-xs">
			WAV, MP3, OGG, FLAC, M4A · MP4, WebM, MOV (audio track only)
		</p>
	</div>
	<input
		bind:this={fileInput}
		type="file"
		accept="audio/*,video/*"
		multiple
		class="hidden"
		onchange={onFileInputChange}
	/>

	{#if jobs.length > 0}
		<!-- Overall batch progress -->
		<div class="bg-muted/40 space-y-2 rounded-lg border p-3">
			<div class="flex items-center justify-between text-xs">
				<span class="text-foreground font-medium">
					{#if batchRunning}
						Processing batch
					{:else if queuedCount > 0}
						Batch queued
					{:else}
						Batch complete
					{/if}
				</span>
				<span class="text-muted-foreground tabular-nums">
					{processedCount} / {jobs.length}
					{jobs.length === 1 ? 'file' : 'files'}
					{#if errorCount > 0}· <span class="text-destructive">{errorCount} error{errorCount === 1 ? '' : 's'}</span>{/if}
				</span>
			</div>
			<Progress value={(processedCount / Math.max(1, jobs.length)) * 100} />

			<div class="flex items-center justify-between pt-1">
				<span class="text-muted-foreground text-[11px]">
					{#if queuedCount > 0}
						{queuedCount} queued
					{:else if batchRunning}
						Finishing last job…
					{:else}
						Done
					{/if}
				</span>
				<div class="flex gap-2">
					{#if jobs.some((j) => j.status === 'queued') && !batchRunning}
						<Button variant="ghost" size="sm" onclick={clearQueued}>Clear queue</Button>
					{/if}
					<Button
						size="sm"
						disabled={!canRun}
						onclick={runBatch}
						title={!workerReady
							? 'Load a model first'
							: queuedCount === 0
								? 'Nothing to process'
								: 'Run batch'}
					>
						{#if batchRunning}
							<Loader2 class="animate-spin" /> Running…
						{:else}
							<Play /> Run batch ({queuedCount})
						{/if}
					</Button>
				</div>
			</div>
		</div>

		<ul class="space-y-2">
			{#each jobs as job (job.id)}
				{@const isActive =
					job.status === 'decoding' || job.status === 'transcribing'}
				<li
					class="bg-card rounded-lg border p-3 text-sm transition-colors {isActive
						? 'border-primary/60 bg-primary/5'
						: ''}"
				>
					<div class="flex items-center gap-3">
						<div class="text-muted-foreground shrink-0">
							{#if job.file.type.startsWith('video/')}
								<FileVideo class="size-4" />
							{:else}
								<FileAudio class="size-4" />
							{/if}
						</div>
						<div class="min-w-0 flex-1">
							<p class="truncate font-medium">{job.file.name}</p>
							<p class="text-muted-foreground text-[11px]">
								{formatBytes(job.file.size)}
							</p>
						</div>
						<div class="text-[11px] shrink-0">
							{#if job.status === 'queued'}
								<span class="text-muted-foreground">Queued</span>
							{:else if job.status === 'decoding'}
								<span class="inline-flex items-center gap-1">
									<Loader2 class="size-3 animate-spin" /> Decoding
								</span>
							{:else if job.status === 'transcribing'}
								<span class="inline-flex items-center gap-1">
									<Loader2 class="size-3 animate-spin" />
									{#if job.totalChunks && job.totalChunks > 1}
										Chunk {(job.chunkIndex ?? 0) + 1}/{job.totalChunks}
									{:else}
										Transcribing
									{/if}
								</span>
							{:else if job.status === 'done'}
								<span
									class="text-emerald-600 inline-flex items-center gap-1 dark:text-emerald-400"
								>
									<CheckCircle2 class="size-3" /> Done
								</span>
							{:else if job.status === 'error'}
								<span class="text-destructive" title={job.error}>Error</span>
							{/if}
						</div>
						{#if !isActive}
							<button
								type="button"
								onclick={() => removeJob(job.id)}
								class="text-muted-foreground hover:text-foreground shrink-0"
								title="Remove"
								aria-label="Remove {job.file.name}"
							>
								<X class="size-4" />
							</button>
						{/if}
					</div>

					{#if isActive || job.status === 'done'}
						<div class="mt-2">
							<Progress
								value={jobProgress(job)}
								class={job.status === 'done' ? 'opacity-60' : ''}
							/>
						</div>
					{/if}
					{#if job.status === 'error' && job.error}
						<p class="text-destructive mt-2 text-[11px] break-words">{job.error}</p>
					{/if}
				</li>
			{/each}
		</ul>
	{/if}
</div>
