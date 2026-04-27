<script lang="ts">
	interface Props {
		stream: MediaStream | null;
		/** Only drives the analyser when true; otherwise renders a flat resting line. */
		active?: boolean;
		class?: string;
	}

	let { stream, active = false, class: className = '' }: Props = $props();

	let canvas: HTMLCanvasElement;

	/** Canvas 2D does not reliably resolve the `currentColor` keyword — most
	 *  browsers treat it as "black". Resolve to an rgb string via the DOM
	 *  instead so the stroke always matches the inherited CSS `color`. */
	function resolveStroke(el: HTMLCanvasElement): string {
		const c = getComputedStyle(el).color;
		// getComputedStyle always returns an rgb()/rgba() string in modern
		// browsers, but in the unlikely case it returns `currentColor` or
		// empty, fall back to a safe neutral.
		return c && c !== 'currentColor' ? c : '#888';
	}

	$effect(() => {
		if (!canvas) return;

		const canvasCtx = canvas.getContext('2d')!;

		const paintIdle = () => {
			canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
			canvasCtx.strokeStyle = resolveStroke(canvas);
			canvasCtx.lineWidth = 1.5;
			canvasCtx.globalAlpha = 0.45;
			canvasCtx.beginPath();
			canvasCtx.moveTo(0, canvas.height / 2);
			canvasCtx.lineTo(canvas.width, canvas.height / 2);
			canvasCtx.stroke();
			canvasCtx.globalAlpha = 1;
		};

		if (!stream || !active) {
			paintIdle();
			return;
		}

		const audioContext = new (window.AudioContext ||
			(window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
		const source = audioContext.createMediaStreamSource(stream);
		const analyser = audioContext.createAnalyser();
		analyser.fftSize = 2048;
		source.connect(analyser);

		const bufferLength = analyser.frequencyBinCount;
		const dataArray = new Uint8Array(bufferLength);

		let raf = 0;
		const draw = () => {
			raf = requestAnimationFrame(draw);
			analyser.getByteTimeDomainData(dataArray);

			canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
			// Re-read every frame so a theme switch is picked up mid-recording.
			canvasCtx.strokeStyle = resolveStroke(canvas);
			canvasCtx.lineWidth = 2.5;
			canvasCtx.lineJoin = 'round';
			canvasCtx.lineCap = 'round';
			canvasCtx.beginPath();

			const sliceWidth = (canvas.width * 1.0) / bufferLength;
			let x = 0;
			for (let i = 0; i < bufferLength; ++i) {
				const v = dataArray[i] / 128.0;
				const y = (v * canvas.height) / 2;
				if (i === 0) canvasCtx.moveTo(x, y);
				else canvasCtx.lineTo(x, y);
				x += sliceWidth;
			}
			canvasCtx.lineTo(canvas.width, canvas.height / 2);
			canvasCtx.stroke();
		};
		draw();

		return () => {
			cancelAnimationFrame(raf);
			source.disconnect();
			audioContext.close();
		};
	});
</script>

<canvas bind:this={canvas} width={720} height={240} class={className}></canvas>
