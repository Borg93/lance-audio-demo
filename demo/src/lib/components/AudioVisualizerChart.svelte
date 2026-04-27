<script lang="ts">
	import { Chart, Svg, Spline } from 'layerchart';
	import { curveMonotoneX } from 'd3-shape';

	interface Props {
		stream: MediaStream | null;
		/** Only drives the analyser when true; otherwise keeps the resting line. */
		active?: boolean;
		class?: string;
	}

	let { stream, active = false, class: className = '' }: Props = $props();

	/** How many points the chart renders. Fewer = smoother, more performant. */
	const N_POINTS = 96;
	/** Target refresh rate (ms between frames). 33 ≈ 30fps. */
	const FRAME_MS = 33;

	type Pt = { x: number; y: number };

	const flat = (): Pt[] => Array.from({ length: N_POINTS }, (_, i) => ({ x: i, y: 0 }));

	let points = $state<Pt[]>(flat());

	$effect(() => {
		if (!stream || !active) {
			points = flat();
			return;
		}

		const audioContext = new (window.AudioContext ||
			(window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
		const source = audioContext.createMediaStreamSource(stream);
		const analyser = audioContext.createAnalyser();
		// Smaller FFT because we only need `N_POINTS` samples.
		analyser.fftSize = 512;
		source.connect(analyser);

		const buf = new Uint8Array(analyser.frequencyBinCount);
		const step = Math.max(1, Math.floor(buf.length / N_POINTS));

		let raf = 0;
		let lastDraw = 0;
		const tick = (t: number) => {
			raf = requestAnimationFrame(tick);
			if (t - lastDraw < FRAME_MS) return;
			lastDraw = t;

			analyser.getByteTimeDomainData(buf);

			const next: Pt[] = new Array(N_POINTS);
			for (let i = 0; i < N_POINTS; i++) {
				// Normalise 0..255 → -1..1, centred on 128.
				next[i] = { x: i, y: (buf[i * step] - 128) / 128 };
			}
			points = next;
		};
		raf = requestAnimationFrame(tick);

		return () => {
			cancelAnimationFrame(raf);
			source.disconnect();
			audioContext.close();
		};
	});
</script>

<div class={className}>
	<Chart
		data={points}
		x="x"
		y="y"
		yDomain={[-1, 1]}
		xDomain={[0, N_POINTS - 1]}
		padding={{ top: 12, bottom: 12, left: 4, right: 4 }}
	>
		<Svg>
			<Spline curve={curveMonotoneX} class="stroke-primary stroke-[2.5] fill-none" />
		</Svg>
	</Chart>
</div>
