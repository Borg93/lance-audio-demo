export type ComputeDevice = 'webgpu' | 'wasm';

/** What the user asked for. `auto` = prefer WebGPU, fall back to WASM. */
export type BackendPreference = 'auto' | 'webgpu' | 'wasm';

export type BackendSpec = {
	key: BackendPreference;
	label: string;
	description: string;
};

export const BACKENDS: Record<BackendPreference, BackendSpec> = {
	auto: {
		key: 'auto',
		label: 'Auto (prefer GPU)',
		description: 'Use WebGPU when available; fall back to WASM CPU otherwise.'
	},
	webgpu: {
		key: 'webgpu',
		label: 'WebGPU (GPU)',
		description: 'Force GPU acceleration. Fastest, but fails if your browser/driver does not support WebGPU.'
	},
	wasm: {
		key: 'wasm',
		label: 'WASM (CPU)',
		description: 'Force the WebAssembly CPU backend. Works everywhere, but noticeably slower on non-trivial models.'
	}
};

/**
 * Probe for a usable WebGPU adapter. Returns 'wasm' if the browser has no
 * `navigator.gpu`, if `requestAdapter()` returns null, or if the call throws.
 */
export async function detectDevice(): Promise<ComputeDevice> {
	if (typeof navigator === 'undefined') return 'wasm';
	const gpu = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu;
	if (!gpu) return 'wasm';
	try {
		const adapter = await gpu.requestAdapter();
		if (adapter) return 'webgpu';
	} catch {
		// fall through to wasm
	}
	return 'wasm';
}

/** Resolve user preference against actual adapter availability.
 *  Throws with a readable message if the preference cannot be satisfied. */
export async function resolveBackend(pref: BackendPreference): Promise<ComputeDevice> {
	if (pref === 'wasm') return 'wasm';
	const detected = await detectDevice();
	if (pref === 'webgpu' && detected !== 'webgpu') {
		throw new Error(
			'WebGPU was requested but no adapter is available. Switch the backend to Auto or WASM, or enable WebGPU in your browser.'
		);
	}
	return detected; // auto, or webgpu-when-available
}
