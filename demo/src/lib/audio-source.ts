export type AudioSource = 'mic' | 'tab';

export type AudioSourceSpec = {
	key: AudioSource;
	label: string;
	description: string;
};

export const AUDIO_SOURCES: Record<AudioSource, AudioSourceSpec> = {
	mic: {
		key: 'mic',
		label: 'Microphone',
		description: 'Record from your default microphone input.'
	},
	tab: {
		key: 'tab',
		label: 'System / Tab audio',
		description:
			'Capture the audio of a browser tab or (on some OSes) the full system. The browser will ask which surface to share — make sure to tick "Share tab audio".'
	}
};

/**
 * Acquire a MediaStream for the requested source. `tab` uses the Screen
 * Capture API; the caller must drop any incidental video tracks.
 */
export async function getAudioStream(source: AudioSource): Promise<MediaStream> {
	if (source === 'mic') {
		return navigator.mediaDevices.getUserMedia({ audio: true });
	}

	// getDisplayMedia with audio only is inconsistent across browsers; Chrome
	// requires at least a video constraint. We strip the video track before
	// returning so downstream code only sees audio.
	const stream = await navigator.mediaDevices.getDisplayMedia({
		audio: true,
		video: true
	});

	const audioTracks = stream.getAudioTracks();
	if (audioTracks.length === 0) {
		stream.getTracks().forEach((t) => t.stop());
		throw new Error(
			'The selected source has no audio track. When the picker appears, choose a tab and tick "Share tab audio".'
		);
	}

	// Discard video tracks — we only need audio.
	stream.getVideoTracks().forEach((t) => {
		t.stop();
		stream.removeTrack(t);
	});

	return stream;
}
