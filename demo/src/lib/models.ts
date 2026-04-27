import type { DataType } from '@huggingface/transformers';

export type ModelKey = 'sv' | 'en';
export type SizeKey = 'tiny' | 'base' | 'small' | 'medium' | 'large';

export type ModelVariant = {
	sizeKey: SizeKey;
	sizeLabel: string;
	/** Hugging Face repo id. */
	modelId: string;
	/** Whisper-large-v3 uses 128 mel bins; everything smaller uses 80. */
	melBins: 80 | 128;
	/** Per-submodule ONNX dtype. */
	dtype: { encoder_model: DataType; decoder_model_merged: DataType };
	/** Rough download size shown in the UI. */
	sizeNote: string;
	/** Optional extra blurb (e.g. "best quality"). */
	tagline?: string;
};

export type LanguagePack = {
	key: ModelKey;
	label: string;
	shortLabel: string;
	/** Whisper language token. */
	language: string;
	defaultSize: SizeKey;
	description: string;
	/** Available sizes, in display order. Not every language ships every size. */
	variants: ModelVariant[];
};

export const LANGUAGES: Record<ModelKey, LanguagePack> = {
	sv: {
		key: 'sv',
		label: 'Swedish',
		shortLabel: 'SV',
		language: 'sv',
		defaultSize: 'base',
		description:
			'KB-Whisper — the National Library of Sweden Swedish finetune, re-exported for Transformers.js by the onnx-community.',
		variants: [
			{
				sizeKey: 'tiny',
				sizeLabel: 'Tiny',
				modelId: 'onnx-community/kb-whisper-tiny-ONNX',
				melBins: 80,
				dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
				sizeNote: '~40 MB',
				tagline: 'Fastest'
			},
			{
				sizeKey: 'base',
				sizeLabel: 'Base',
				modelId: 'onnx-community/kb-whisper-base-ONNX',
				melBins: 80,
				dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
				sizeNote: '~80 MB',
				tagline: 'Balanced'
			},
			{
				sizeKey: 'small',
				sizeLabel: 'Small',
				modelId: 'onnx-community/kb-whisper-small-ONNX',
				melBins: 80,
				dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
				sizeNote: '~160 MB'
			},
			{
				sizeKey: 'medium',
				sizeLabel: 'Medium',
				modelId: 'onnx-community/kb-whisper-medium-ONNX',
				melBins: 80,
				dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
				sizeNote: '~300 MB'
			},
			{
				sizeKey: 'large',
				sizeLabel: 'Large',
				modelId: 'onnx-community/kb-whisper-large-ONNX',
				// large-v3 architecture → 128 mel bins.
				melBins: 128,
				dtype: { encoder_model: 'fp16', decoder_model_merged: 'q4f16' },
				sizeNote: '~1 GB',
				tagline: 'Best quality'
			}
		]
	},
	en: {
		key: 'en',
		label: 'English',
		shortLabel: 'EN',
		language: 'en',
		defaultSize: 'base',
		description:
			"OpenAI Whisper, ONNX-exported for browser inference. Multilingual at heart but tuned here for English.",
		variants: [
			{
				sizeKey: 'tiny',
				sizeLabel: 'Tiny',
				modelId: 'onnx-community/whisper-tiny',
				melBins: 80,
				dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
				sizeNote: '~40 MB',
				tagline: 'Fastest'
			},
			{
				sizeKey: 'base',
				sizeLabel: 'Base',
				modelId: 'onnx-community/whisper-base',
				melBins: 80,
				dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
				sizeNote: '~80 MB',
				tagline: 'Balanced'
			},
			{
				sizeKey: 'small',
				sizeLabel: 'Small',
				modelId: 'onnx-community/whisper-small',
				melBins: 80,
				dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
				sizeNote: '~240 MB'
			},
			{
				sizeKey: 'medium',
				sizeLabel: 'Medium',
				modelId: 'onnx-community/whisper-medium',
				melBins: 80,
				dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
				sizeNote: '~760 MB'
			}
		]
	}
};

export const DEFAULT_LANGUAGE: ModelKey = 'sv';

export function findVariant(langKey: ModelKey, sizeKey: SizeKey): ModelVariant {
	const pack = LANGUAGES[langKey];
	return pack.variants.find((v) => v.sizeKey === sizeKey) ?? pack.variants[0];
}
