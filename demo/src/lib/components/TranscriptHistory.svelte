<script lang="ts">
	import FileText from 'lucide-svelte/icons/file-text';
	import FileJson from 'lucide-svelte/icons/file-json';
	import Captions from 'lucide-svelte/icons/captions';
	import AudioLines from 'lucide-svelte/icons/audio-lines';
	import Trash2 from 'lucide-svelte/icons/trash-2';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import Loader2 from 'lucide-svelte/icons/loader-2';
	import { Button } from '$lib/components/ui/button';
	import {
		audioExtension,
		downloadBlob,
		entryBasename,
		formatTimestamp,
		formatWallClock,
		toJson,
		toSrt,
		toTxt,
		type TranscriptEntry
	} from '$lib/transcript';
	import { blobToWav } from '$lib/wav';

	function formatBytes(n: number): string {
		if (n === 0) return '0 B';
		const i = Math.floor(Math.log(n) / Math.log(1024));
		const units = ['B', 'kB', 'MB', 'GB'];
		return (n / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1) + ' ' + units[i];
	}

	interface Props {
		entries: TranscriptEntry[];
		onDelete: (id: string) => void;
	}

	let { entries, onDelete }: Props = $props();

	let expanded = $state<Set<string>>(new Set());
	let encodingWav = $state<Set<string>>(new Set());

	function toggle(id: string) {
		const next = new Set(expanded);
		if (next.has(id)) next.delete(id);
		else next.add(id);
		expanded = next;
	}

	function setEncoding(id: string, busy: boolean) {
		const next = new Set(encodingWav);
		if (busy) next.add(id);
		else next.delete(id);
		encodingWav = next;
	}

	function downloadTxt(entry: TranscriptEntry) {
		downloadBlob(new Blob([toTxt(entry)], { type: 'text/plain' }), `${entryBasename(entry)}.txt`);
	}
	function downloadJson(entry: TranscriptEntry) {
		downloadBlob(
			new Blob([toJson(entry)], { type: 'application/json' }),
			`${entryBasename(entry)}.json`
		);
	}
	function downloadSrt(entry: TranscriptEntry) {
		downloadBlob(new Blob([toSrt(entry)], { type: 'text/plain' }), `${entryBasename(entry)}.srt`);
	}
	async function downloadWav(entry: TranscriptEntry) {
		if (encodingWav.has(entry.id)) return;
		if (entry.audioBlob.size === 0) {
			alert('This entry has no audio data.');
			return;
		}
		setEncoding(entry.id, true);
		try {
			const wav = await blobToWav(entry.audioBlob);
			downloadBlob(wav, `${entryBasename(entry)}.wav`);
		} catch (err) {
			console.error('WAV encode failed:', err);
			alert(
				`Couldn't encode WAV: ${err instanceof Error ? err.message : String(err)}. ` +
					`The original audio was recorded as ${entry.audioType}.`
			);
		} finally {
			setEncoding(entry.id, false);
		}
	}

	function downloadRaw(entry: TranscriptEntry) {
		if (entry.audioBlob.size === 0) {
			alert('This entry has no audio data.');
			return;
		}
		downloadBlob(entry.audioBlob, `${entryBasename(entry)}.${audioExtension(entry.audioType)}`);
	}
</script>

{#if entries.length > 0}
	<section class="flex flex-col gap-3">
		<header class="flex items-baseline justify-between">
			<h2 class="text-sm font-semibold">Session history</h2>
			<span class="text-muted-foreground text-xs">
				{entries.length} saved · {entries.length === 1 ? 'session' : 'sessions'}
			</span>
		</header>

		<ul class="space-y-2">
			{#each entries as entry (entry.id)}
				{@const isOpen = expanded.has(entry.id)}
				<li class="bg-card rounded-lg border">
					<button
						type="button"
						onclick={() => toggle(entry.id)}
						class="hover:bg-muted/30 flex w-full items-start gap-3 rounded-lg p-3 text-left transition-colors"
						aria-expanded={isOpen}
					>
						<div class="flex-1 space-y-1 min-w-0">
							<div class="text-muted-foreground flex items-center gap-2 text-[11px]">
								<span>{formatWallClock(entry.startedAt)}</span>
								<span>·</span>
								<span class="tabular-nums">{formatTimestamp(entry.durationMs)}</span>
								<span>·</span>
								<span class="font-mono uppercase">{entry.language}</span>
								<span>·</span>
								<span class="truncate">{entry.modelId}</span>
							</div>
							<p class="text-sm {isOpen ? '' : 'line-clamp-2'}">
								{#if entry.text}
									{entry.text}
								{:else}
									<em class="text-muted-foreground">(empty transcript)</em>
								{/if}
							</p>
						</div>
						<ChevronDown
							class="text-muted-foreground mt-1 size-4 shrink-0 transition-transform {isOpen
								? 'rotate-180'
								: ''}"
						/>
					</button>

					{#if isOpen}
						<div class="border-t p-3">
							{#if entry.segments.length > 0}
								<div class="mb-3 max-h-60 overflow-y-auto rounded-md border">
									<table class="w-full text-xs">
										<tbody>
											{#each entry.segments as seg, i (i)}
												<tr class="border-b last:border-b-0">
													<td
														class="text-muted-foreground w-20 px-2 py-1 font-mono tabular-nums"
													>
														{formatTimestamp(seg.start * 1000)}
													</td>
													<td class="px-2 py-1">{seg.text}</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
							{:else}
								<p class="text-muted-foreground mb-3 text-xs">
									No segment timestamps (finalisation was skipped or returned a single block).
								</p>
							{/if}

							<div class="text-muted-foreground mb-2 text-[11px]">
								Audio size:
								<span class="text-foreground font-mono tabular-nums">
									{formatBytes(entry.audioBlob.size)}
								</span>
								{#if entry.audioBlob.size === 0}
									<span class="text-destructive ml-2">
										— no audio captured; downloads will be empty
									</span>
								{/if}
							</div>
							<div class="flex flex-wrap gap-2">
								<Button variant="outline" size="sm" onclick={() => downloadTxt(entry)}>
									<FileText /> .txt
								</Button>
								<Button variant="outline" size="sm" onclick={() => downloadSrt(entry)}>
									<Captions /> .srt
								</Button>
								<Button variant="outline" size="sm" onclick={() => downloadJson(entry)}>
									<FileJson /> .json
								</Button>
								<Button
									variant="outline"
									size="sm"
									disabled={encodingWav.has(entry.id) || entry.audioBlob.size === 0}
									onclick={() => downloadWav(entry)}
								>
									{#if encodingWav.has(entry.id)}
										<Loader2 class="animate-spin" />
									{:else}
										<AudioLines />
									{/if}
									.wav
								</Button>
								<Button
									variant="outline"
									size="sm"
									disabled={entry.audioBlob.size === 0}
									onclick={() => downloadRaw(entry)}
									title="Raw recorder output ({entry.audioType})"
								>
									<AudioLines />
									.{audioExtension(entry.audioType)}
								</Button>
								<Button
									variant="ghost"
									size="sm"
									class="text-destructive ml-auto"
									onclick={() => onDelete(entry.id)}
								>
									<Trash2 /> Delete
								</Button>
							</div>
						</div>
					{/if}
				</li>
			{/each}
		</ul>
	</section>
{/if}
