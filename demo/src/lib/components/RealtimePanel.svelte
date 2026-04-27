<script lang="ts">
	import { onDestroy } from 'svelte';
	import Mic from 'lucide-svelte/icons/mic';
	import RotateCcw from 'lucide-svelte/icons/rotate-ccw';
	import Loader2 from 'lucide-svelte/icons/loader-2';
	import CircleX from 'lucide-svelte/icons/circle-x';
	import Gauge from 'lucide-svelte/icons/gauge';
	import Square from 'lucide-svelte/icons/square';

	import AudioVisualizer from './AudioVisualizerChart.svelte';
	import { Button } from './ui/button';
	import { Progress } from './ui/progress';
	import * as Select from './ui/select';
	import { ShimmerButton } from './magic';

	import { AUDIO_SOURCES, getAudioStream, type AudioSource } from '$lib/audio-source';
	import type { LanguagePack, ModelVariant } from '$lib/models';
	import type { TranscriptEntry, TranscriptSegment } from '$lib/transcript';

	interface Props {
		worker: Worker | null;
		workerReady: boolean;
		modelSpec: ModelVariant;
		languagePack: LanguagePack;
		onEntry: (entry: TranscriptEntry) => void;
	}

	let { worker, workerReady, modelSpec, languagePack, onEntry }: Props = $props();

	const WHISPER_SAMPLING_RATE = 16_000;
	const MAX_AUDIO_LENGTH = 30;
	const MAX_SAMPLES = WHISPER_SAMPLING_RATE * MAX_AUDIO_LENGTH;

	let audioSource = $state<AudioSource>('mic');
	const sourceSpec = $derived(AUDIO_SOURCES[audioSource]);

	let text = $state<string | string[]>('');
	let tps = $state<number | null>(null);
	let errorMessage = $state('');

	let recording = $state(false);
	let isProcessing = $state(false);
	let chunks = $state<Blob[]>([]);
	let stream = $state<MediaStream | null>(null);
	let finalizing = $state(false);
	let finalizeChunkIndex = $state(0);
	let finalizeTotalChunks = $state(0);

	let recorder: MediaRecorder | null = null;
	let audioContext: AudioContext | null = null;
	let sessionChunks: Blob[] = [];
	let sessionStartedAt = 0;
	let pendingEntry: Pick<TranscriptEntry, 'audioBlob' | 'audioType' | 'startedAt' | 'durationMs'> | null = null;

	const displayText = $derived(Array.isArray(text) ? text.join(' ') : text);

	// Attach / reattach the worker listener whenever the worker changes.
	$effect(() => {
		if (!worker) return;
		const handler = (e: MessageEvent) => {
			const d = e.data;
			// Ignore messages tagged with a jobId — those belong to BatchPanel.
			if (d.jobId !== undefined) return;
			switch (d.status) {
				case 'start':
					isProcessing = true;
					recorder?.requestData();
					break;
				case 'update':
					tps = d.tps;
					break;
				case 'complete':
					isProcessing = false;
					text = d.output;
					break;
				case 'finalize-start':
					finalizing = true;
					finalizeChunkIndex = 0;
					finalizeTotalChunks = 0;
					break;
				case 'finalize-progress':
					finalizeChunkIndex = d.chunkIndex ?? 0;
					finalizeTotalChunks = d.totalChunks ?? 0;
					break;
				case 'finalized':
					finalizing = false;
					finalizeChunkIndex = 0;
					finalizeTotalChunks = 0;
					handleFinalized(d.text as string, d.segments as TranscriptSegment[]);
					break;
				case 'error':
					errorMessage = d.data;
					finalizing = false;
					break;
			}
		};
		worker.addEventListener('message', handler);
		return () => worker?.removeEventListener('message', handler);
	});

	onDestroy(() => {
		recorder?.stop();
		recorder = null;
		stream?.getTracks().forEach((t) => t.stop());
		stream = null;
		audioContext?.close();
		audioContext = null;
	});

	// Live-loop: ship rolling-window audio to the worker for preview.
	$effect(() => {
		const currentRecording = recording;
		const currentIsProcessing = isProcessing;
		const currentChunks = chunks;

		if (!recorder) return;
		if (!workerReady) return;
		if (!currentRecording) return;
		if (currentIsProcessing) return;

		if (currentChunks.length > 0) {
			const blob = new Blob(currentChunks, { type: recorder.mimeType });
			const fileReader = new FileReader();
			fileReader.onloadend = async () => {
				const arrayBuffer = fileReader.result as ArrayBuffer;
				const decoded = await audioContext!.decodeAudioData(arrayBuffer);
				const channel = decoded.getChannelData(0);
				const audio =
					channel.length > MAX_SAMPLES ? channel.slice(-MAX_SAMPLES) : channel.slice();
				worker?.postMessage({ type: 'generate', data: { audio } }, [audio.buffer]);
			};
			fileReader.readAsArrayBuffer(blob);
		} else {
			recorder.requestData();
		}
	});

	function handleFinalized(finalText: string, segments: TranscriptSegment[]) {
		if (!pendingEntry) return;
		onEntry({
			id: crypto.randomUUID(),
			startedAt: pendingEntry.startedAt,
			durationMs: pendingEntry.durationMs,
			modelId: modelSpec.modelId,
			language: languagePack.language,
			audioSource,
			text: finalText.trim(),
			segments,
			audioBlob: pendingEntry.audioBlob,
			audioType: pendingEntry.audioType
		});
		pendingEntry = null;
	}

	function onSourceChange(next: string | undefined) {
		if (!next) return;
		if (recording) stopTranscribing();
		audioSource = next as AudioSource;
	}

	async function startTranscribing() {
		if (!workerReady) return;
		text = '';
		tps = null;
		chunks = [];
		sessionChunks = [];
		errorMessage = '';

		try {
			const s = await getAudioStream(audioSource);
			stream = s;
			audioContext ??= new AudioContext({ sampleRate: WHISPER_SAMPLING_RATE });
			recorder = new MediaRecorder(s);

			recorder.onstart = () => {
				recording = true;
				chunks = [];
				sessionChunks = [];
				sessionStartedAt = Date.now();
			};
			recorder.ondataavailable = (e) => {
				if (e.data.size > 0) {
					chunks = [...chunks, e.data];
					sessionChunks = [...sessionChunks, e.data];
				} else {
					setTimeout(() => recorder?.requestData(), 25);
				}
			};
			recorder.onstop = async () => {
				recording = false;
				const stoppedAt = Date.now();
				stream?.getTracks().forEach((t) => t.stop());
				stream = null;

				if (sessionChunks.length === 0 || !worker) return;

				const mimeType = recorder?.mimeType ?? 'audio/webm';
				const audioBlob = new Blob(sessionChunks, { type: mimeType });

				try {
					const arrayBuffer = await audioBlob.arrayBuffer();
					const decoded = await audioContext!.decodeAudioData(arrayBuffer.slice(0));
					const audio = decoded.getChannelData(0).slice();

					pendingEntry = {
						startedAt: sessionStartedAt,
						durationMs: stoppedAt - sessionStartedAt,
						audioBlob,
						audioType: mimeType
					};

					worker.postMessage({ type: 'finalize', data: { audio } }, [audio.buffer]);
				} catch (err) {
					pendingEntry = null;
					console.error('finalization decode error:', err);
					if (displayText) {
						onEntry({
							id: crypto.randomUUID(),
							startedAt: sessionStartedAt,
							durationMs: stoppedAt - sessionStartedAt,
							modelId: modelSpec.modelId,
							language: languagePack.language,
							audioSource,
							text: displayText.trim(),
							segments: [],
							audioBlob,
							audioType: mimeType
						});
					}
				}
			};

			s.getAudioTracks().forEach((track) => {
				track.addEventListener('ended', () => recorder?.stop());
			});

			recorder.start();
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			errorMessage = `Couldn't open ${sourceSpec.label.toLowerCase()}: ${msg}`;
			console.error('audio source error:', err);
		}
	}

	function stopTranscribing() {
		recorder?.stop();
	}

	function clearTranscript() {
		text = '';
		tps = null;
	}
</script>

<div class="flex flex-col gap-4">
	{#if errorMessage}
		<div
			class="border-destructive/40 bg-destructive/10 text-destructive flex items-start gap-2 rounded-md border p-3 text-sm"
		>
			<CircleX class="mt-0.5 size-4 shrink-0" />
			<p class="text-destructive/80 break-words">{errorMessage}</p>
		</div>
	{/if}

	<div class="bg-muted/30 text-foreground overflow-hidden rounded-lg border">
		<AudioVisualizer class="h-[140px] w-full" {stream} active={recording} />
	</div>

	<div class="relative">
		<p
			class="bg-background/60 overflow-wrap-anywhere min-h-[120px] w-full overflow-y-auto rounded-lg border p-4 text-base leading-relaxed"
		>
			{#if recording}
				{displayText || `Listening in ${languagePack.label}…`}
			{:else if displayText}
				{displayText}
			{:else if !workerReady}
				<span class="text-muted-foreground">Load a model above to enable realtime transcription.</span>
			{:else}
				<span class="text-muted-foreground">
					Press <span class="text-foreground font-medium">Start transcribing</span> and speak in {languagePack.label}.
				</span>
			{/if}
		</p>
		{#if tps}
			<span
				class="text-muted-foreground absolute right-3 bottom-3 flex items-center gap-1 text-xs tabular-nums"
			>
				<Gauge class="size-3" />
				{tps.toFixed(2)} tok/s
			</span>
		{/if}
	</div>

	<div class="flex flex-col items-center gap-3 md:flex-row md:justify-between">
		<div class="order-2 flex flex-wrap items-center gap-2 md:order-1">
			<Select.Root
				type="single"
				value={audioSource}
				onValueChange={onSourceChange}
				disabled={recording}
			>
				<Select.Trigger class="h-8 w-[180px] text-xs">
					<span class="text-muted-foreground mr-1">Source</span>
					<span class="font-medium">{sourceSpec.label}</span>
				</Select.Trigger>
				<Select.Content>
					{#each Object.values(AUDIO_SOURCES) as spec (spec.key)}
						<Select.Item value={spec.key} label={spec.label} />
					{/each}
				</Select.Content>
			</Select.Root>
			{#if displayText || tps}
				<Button variant="ghost" size="sm" onclick={clearTranscript}>
					<RotateCcw /> Clear
				</Button>
			{/if}
		</div>

		<div class="order-1 md:order-2 flex flex-col items-center gap-1">
			{#if finalizing}
				<div class="bg-muted/40 w-full max-w-md space-y-2 rounded-xl border px-5 py-3">
					<div class="flex items-center gap-2 text-sm">
						<Loader2 class="animate-spin size-4" />
						<span>Finalizing transcript & timestamps…</span>
						{#if finalizeTotalChunks > 1}
							<span class="text-muted-foreground ml-auto text-xs tabular-nums">
								Chunk {finalizeChunkIndex + 1}/{finalizeTotalChunks}
							</span>
						{/if}
					</div>
					{#if finalizeTotalChunks > 0}
						<Progress
							value={((finalizeChunkIndex + 1) / Math.max(1, finalizeTotalChunks)) * 100}
						/>
					{/if}
				</div>
			{:else if !recording}
				<ShimmerButton
					onclick={startTranscribing}
					disabled={!workerReady}
					background="linear-gradient(135deg, oklch(0.488 0.243 264.376), oklch(0.577 0.245 27.325))"
					shimmerColor="#ffffff"
					shimmerDuration="2.5s"
					class="min-w-[240px] px-8"
				>
					<Mic class="size-4" />
					Start transcribing
				</ShimmerButton>
			{:else}
				<button
					type="button"
					onclick={stopTranscribing}
					class="bg-destructive text-destructive-foreground ring-destructive/40 ring-offset-background hover:bg-destructive/90 relative inline-flex min-w-[240px] items-center justify-center gap-2 rounded-full px-8 py-3 text-sm font-medium ring-4 ring-offset-2 transition-all focus-visible:outline-hidden"
					aria-pressed="true"
				>
					<span class="relative flex size-2.5">
						<span
							class="bg-destructive-foreground absolute inline-flex size-full animate-ping rounded-full opacity-75"
						></span>
						<span class="bg-destructive-foreground relative inline-flex size-2.5 rounded-full"
						></span>
					</span>
					<Square class="size-4 fill-current" />
					Stop transcribing
				</button>
				<span class="text-destructive flex items-center gap-1.5 text-xs font-medium">
					<span class="relative flex size-1.5">
						<span
							class="bg-destructive absolute inline-flex size-full animate-ping rounded-full opacity-75"
						></span>
						<span class="bg-destructive relative inline-flex size-1.5 rounded-full"></span>
					</span>
					REC · {sourceSpec.label} · {languagePack.label}
				</span>
			{/if}
		</div>
	</div>
</div>
