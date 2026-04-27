export type TranscriptSegment = {
	text: string;
	/** Seconds from the start of the session. */
	start: number;
	/** Seconds from the start of the session. */
	end: number;
};

export type TranscriptEntry = {
	id: string;
	startedAt: number; // Date.now()
	durationMs: number;
	modelId: string;
	language: string;
	audioSource: 'mic' | 'tab';
	text: string;
	segments: TranscriptSegment[]; // [] when timestamps are unavailable
	audioBlob: Blob;
	audioType: string; // MIME of audioBlob
};

export function formatTimestamp(ms: number): string {
	const total = Math.round(ms / 1000);
	const s = total % 60;
	const m = Math.floor(total / 60) % 60;
	const h = Math.floor(total / 3600);
	const pad = (n: number, w = 2) => n.toString().padStart(w, '0');
	return h > 0 ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

export function formatWallClock(ts: number): string {
	return new Date(ts).toLocaleString(undefined, {
		year: 'numeric',
		month: 'short',
		day: '2-digit',
		hour: '2-digit',
		minute: '2-digit'
	});
}

/** Parse Whisper-style `<|0.00|>text<|2.50|>` segments out of a decoded string. */
export function parseWhisperTimestamps(raw: string): TranscriptSegment[] {
	const out: TranscriptSegment[] = [];
	const pattern = /<\|(\d+(?:\.\d+)?)\|>([^<]*?)<\|(\d+(?:\.\d+)?)\|>/g;
	let match: RegExpExecArray | null;
	while ((match = pattern.exec(raw)) !== null) {
		const start = parseFloat(match[1]);
		const end = parseFloat(match[3]);
		const text = match[2].trim();
		if (!text) continue;
		out.push({ start, end, text });
	}
	return out;
}

function srtTimestamp(seconds: number): string {
	const ms = Math.floor(seconds * 1000);
	const h = Math.floor(ms / 3_600_000);
	const m = Math.floor((ms % 3_600_000) / 60_000);
	const s = Math.floor((ms % 60_000) / 1000);
	const milli = ms % 1000;
	const pad = (n: number, w = 2) => n.toString().padStart(w, '0');
	return `${pad(h)}:${pad(m)}:${pad(s)},${milli.toString().padStart(3, '0')}`;
}

/** Generate an SRT string from segments. Falls back to a single cue covering
 *  the whole duration when segments are empty. */
export function toSrt(entry: TranscriptEntry): string {
	const segments =
		entry.segments.length > 0
			? entry.segments
			: [{ start: 0, end: entry.durationMs / 1000, text: entry.text }];

	return segments
		.map(
			(seg, i) =>
				`${i + 1}\n${srtTimestamp(seg.start)} --> ${srtTimestamp(seg.end)}\n${seg.text}\n`
		)
		.join('\n');
}

export function toTxt(entry: TranscriptEntry): string {
	return entry.text.trim() + '\n';
}

export function toJson(entry: TranscriptEntry): string {
	// Strip the audio Blob from the JSON (separate download).
	const { audioBlob: _audioBlob, audioType: _audioType, ...rest } = entry;
	return JSON.stringify(rest, null, 2);
}

export function downloadBlob(blob: Blob, filename: string) {
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	// Give the browser a tick to start the download before we revoke.
	setTimeout(() => URL.revokeObjectURL(url), 500);
}

export function audioExtension(mime: string): string {
	if (mime.includes('webm')) return 'webm';
	if (mime.includes('ogg')) return 'ogg';
	if (mime.includes('mp4')) return 'mp4';
	if (mime.includes('wav')) return 'wav';
	return 'audio';
}

export function entryBasename(entry: TranscriptEntry): string {
	const d = new Date(entry.startedAt);
	const pad = (n: number) => n.toString().padStart(2, '0');
	return `transcript-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}-${entry.language}`;
}
