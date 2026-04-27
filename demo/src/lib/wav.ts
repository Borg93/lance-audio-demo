/**
 * PCM-WAV encoder. Takes an AudioBuffer and returns a 16-bit little-endian
 * RIFF/WAVE Blob with interleaved samples — the universally-supported format
 * for DAWs, whisper.cpp, ffmpeg, and every desktop audio player.
 */
export function encodeWav(audioBuffer: AudioBuffer): Blob {
	const numChannels = audioBuffer.numberOfChannels;
	const sampleRate = audioBuffer.sampleRate;
	const bytesPerSample = 2;
	const dataLength = audioBuffer.length * numChannels * bytesPerSample;
	const bufferLength = 44 + dataLength;

	const ab = new ArrayBuffer(bufferLength);
	const view = new DataView(ab);

	// RIFF header
	writeString(view, 0, 'RIFF');
	view.setUint32(4, bufferLength - 8, true);
	writeString(view, 8, 'WAVE');

	// fmt  subchunk
	writeString(view, 12, 'fmt ');
	view.setUint32(16, 16, true); // subchunk size
	view.setUint16(20, 1, true); // PCM format
	view.setUint16(22, numChannels, true);
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
	view.setUint16(32, numChannels * bytesPerSample, true);
	view.setUint16(34, 16, true); // bit depth

	// data subchunk
	writeString(view, 36, 'data');
	view.setUint32(40, dataLength, true);

	// Interleave and write samples.
	const channels: Float32Array[] = [];
	for (let c = 0; c < numChannels; c++) channels.push(audioBuffer.getChannelData(c));

	let offset = 44;
	for (let i = 0; i < audioBuffer.length; i++) {
		for (let c = 0; c < numChannels; c++) {
			const s = Math.max(-1, Math.min(1, channels[c][i]));
			view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
			offset += 2;
		}
	}

	return new Blob([ab], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, str: string) {
	for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

/** Decode a compressed audio Blob (webm/opus/ogg/…) → AudioBuffer → WAV Blob. */
export async function blobToWav(blob: Blob): Promise<Blob> {
	const ctx = new (window.AudioContext ||
		(window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
	try {
		const arrayBuffer = await blob.arrayBuffer();
		const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
		return encodeWav(audioBuffer);
	} finally {
		await ctx.close();
	}
}
