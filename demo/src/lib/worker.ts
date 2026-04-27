/// <reference lib="webworker" />
import {
	AutoTokenizer,
	AutoProcessor,
	WhisperForConditionalGeneration,
	TextStreamer,
	Tensor,
	env,
	full,
	type DataType
} from '@huggingface/transformers';
import { resolveBackend, type BackendPreference, type ComputeDevice } from './webgpu';
import { parseWhisperTimestamps, type TranscriptSegment } from './transcript';

type LoadSpec = {
	modelId: string;
	language: string;
	melBins: 80 | 128;
	dtype: { encoder_model: DataType; decoder_model_merged: DataType };
};

const MAX_NEW_TOKENS = 64;
/** Whisper's context length — used for the full finalization pass. */
const WHISPER_MAX_NEW_TOKENS = 445;
const MEL_FRAMES = 3000;
/** Whisper's native window length (seconds) and equivalent PCM samples at 16 kHz. */
const CHUNK_SECONDS = 30;
const CHUNK_SAMPLES = CHUNK_SECONDS * 16_000;

// Keep browser-cache on (v4 default, but explicit for clarity) so weights
// are reused on revisits and when switching back to a previously-loaded model.
env.useBrowserCache = true;

type Tokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
type Processor = Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>;
type Model = Awaited<ReturnType<typeof WhisperForConditionalGeneration.from_pretrained>>;

class Pipeline {
	static spec: LoadSpec | null = null;
	static tokenizer: Promise<Tokenizer> | null = null;
	static processor: Promise<Processor> | null = null;
	static model: Promise<Model> | null = null;
	static device: ComputeDevice = 'wasm';

	static async getInstance(spec: LoadSpec, progress_callback?: (x: unknown) => void) {
		this.spec = spec;
		this.tokenizer ??= AutoTokenizer.from_pretrained(spec.modelId, { progress_callback });
		this.processor ??= AutoProcessor.from_pretrained(spec.modelId, { progress_callback });
		this.model ??= WhisperForConditionalGeneration.from_pretrained(spec.modelId, {
			dtype: spec.dtype,
			device: this.device,
			progress_callback
		});
		return Promise.all([this.tokenizer, this.processor, this.model]);
	}
}

let processing = false;

async function generate({ audio }: { audio: Float32Array }) {
	if (processing) return;
	processing = true;

	self.postMessage({ status: 'start' });

	try {
		if (!Pipeline.spec) throw new Error('generate() called before load()');
		const [tokenizer, processor, model] = await Pipeline.getInstance(Pipeline.spec);

		let startTime: number | undefined;
		let numTokens = 0;
		let tps: number | undefined;

		const token_callback_function = () => {
			startTime ??= performance.now();
			if (numTokens++ > 0) {
				tps = (numTokens / (performance.now() - startTime)) * 1000;
			}
		};

		const callback_function = (output: string) => {
			self.postMessage({ status: 'update', output, tps, numTokens });
		};

		const streamer = new TextStreamer(tokenizer, {
			skip_prompt: true,
			skip_special_tokens: true,
			callback_function,
			token_callback_function
		});

		const inputs = await processor(audio);

		const outputs = (await model.generate({
			...inputs,
			max_new_tokens: MAX_NEW_TOKENS,
			language: Pipeline.spec.language,
			// Explicit task — without it whisper-large-v3 finetunes can route
			// through the translate head and produce gibberish.
			task: 'transcribe',
			streamer
		})) as Tensor;

		const decoded = tokenizer.batch_decode(outputs, { skip_special_tokens: true });

		self.postMessage({ status: 'complete', output: decoded });
	} catch (err) {
		self.postMessage({
			status: 'error',
			data: `Inference failed: ${err instanceof Error ? err.message : String(err)}`
		});
	} finally {
		processing = false;
	}
}

async function load(spec: LoadSpec, backendPref: BackendPreference = 'auto') {
	if (!spec?.modelId) {
		self.postMessage({ status: 'error', data: 'No model id supplied.' });
		return;
	}

	self.postMessage({ status: 'loading', data: 'Selecting compute backend…' });

	try {
		Pipeline.device = await resolveBackend(backendPref);
	} catch (err) {
		self.postMessage({
			status: 'error',
			data: err instanceof Error ? err.message : String(err)
		});
		return;
	}
	self.postMessage({ status: 'device', device: Pipeline.device });

	self.postMessage({ status: 'loading', data: `Loading ${spec.modelId}…` });

	let model: Model;
	try {
		[, , model] = await Pipeline.getInstance(spec, (x) => self.postMessage(x));
	} catch (err) {
		self.postMessage({
			status: 'error',
			data: `Download failed: ${err instanceof Error ? err.message : String(err)}`
		});
		return;
	}

	self.postMessage({ status: 'loading', data: 'Compiling shaders and warming up model…' });

	try {
		await model.generate({
			input_features: full([1, spec.melBins, MEL_FRAMES], 0.0),
			max_new_tokens: 1
		});
	} catch (err) {
		self.postMessage({
			status: 'error',
			data: `Warm-up failed: ${err instanceof Error ? err.message : String(err)}`
		});
		return;
	}

	self.postMessage({ status: 'ready' });
}

async function finalize({ audio, jobId }: { audio: Float32Array; jobId?: string }) {
	self.postMessage({ status: 'finalize-start', jobId });

	try {
		if (!Pipeline.spec) throw new Error('finalize() called before load()');
		const [tokenizer, processor, model] = await Pipeline.getInstance(Pipeline.spec);

		const totalChunks = Math.max(1, Math.ceil(audio.length / CHUNK_SAMPLES));
		let fullText = '';
		const fullSegments: TranscriptSegment[] = [];

		for (let i = 0; i < totalChunks; i++) {
			self.postMessage({
				status: 'finalize-progress',
				jobId,
				chunkIndex: i,
				totalChunks
			});

			const chunk = audio.subarray(
				i * CHUNK_SAMPLES,
				Math.min((i + 1) * CHUNK_SAMPLES, audio.length)
			);
			const offsetSec = i * CHUNK_SECONDS;

			const inputs = await processor(chunk);
			const outputs = (await model.generate({
				...inputs,
				max_new_tokens: WHISPER_MAX_NEW_TOKENS,
				language: Pipeline.spec.language,
				task: 'transcribe',
				return_timestamps: true
			})) as Tensor;

			const [plainText] = tokenizer.batch_decode(outputs, { skip_special_tokens: true });
			const [rawWithTimestamps] = tokenizer.batch_decode(outputs, {
				skip_special_tokens: false
			});

			fullText += (fullText ? ' ' : '') + plainText.trim();
			for (const seg of parseWhisperTimestamps(rawWithTimestamps)) {
				fullSegments.push({
					start: seg.start + offsetSec,
					end: seg.end + offsetSec,
					text: seg.text
				});
			}
		}

		self.postMessage({
			status: 'finalized',
			jobId,
			text: fullText,
			segments: fullSegments
		});
	} catch (err) {
		self.postMessage({
			status: 'error',
			jobId,
			data: `Finalization failed: ${err instanceof Error ? err.message : String(err)}`
		});
	}
}

self.addEventListener('message', async (e: MessageEvent) => {
	const { type, data } = e.data;
	switch (type) {
		case 'load':
			load(data.spec, data.backendPref);
			break;
		case 'generate':
			generate(data);
			break;
		case 'finalize':
			finalize(data);
			break;
	}
});
