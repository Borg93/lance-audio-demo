/**
 * Decode any media file (audio or video container the browser understands) to
 * a mono Float32 PCM buffer at `targetSampleRate` (default 16 kHz — Whisper's
 * expected input rate).
 *
 * Supports WAV, MP3, OGG/Opus, FLAC, M4A/AAC, plus video containers like MP4,
 * WebM, MOV whenever the browser's `decodeAudioData` can read them. Video
 * tracks are ignored; only the audio track is extracted.
 */
export async function decodeMediaFile(
	file: Blob,
	targetSampleRate = 16_000
): Promise<Float32Array> {
	const arrayBuffer = await file.arrayBuffer();

	// Use a default-rate AudioContext first — some browsers refuse to decode
	// directly into a non-standard rate context. We resample afterwards.
	const decodeCtx = new (window.AudioContext ||
		(window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
	let audioBuffer: AudioBuffer;
	try {
		audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
	} finally {
		await decodeCtx.close();
	}

	// Already mono at target rate? Skip the offline resample pass.
	if (audioBuffer.sampleRate === targetSampleRate && audioBuffer.numberOfChannels === 1) {
		return audioBuffer.getChannelData(0).slice();
	}

	const offline = new OfflineAudioContext(
		1, // downmix to mono
		Math.max(1, Math.ceil(audioBuffer.duration * targetSampleRate)),
		targetSampleRate
	);
	const source = offline.createBufferSource();
	source.buffer = audioBuffer;
	source.connect(offline.destination);
	source.start(0);
	const resampled = await offline.startRendering();
	return resampled.getChannelData(0).slice();
}

/** Loose check for what the UI will accept. Actual support depends on browser. */
export function isMediaFile(file: File): boolean {
	return file.type.startsWith('audio/') || file.type.startsWith('video/');
}
