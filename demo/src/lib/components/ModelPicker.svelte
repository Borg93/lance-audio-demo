<script lang="ts">
	import * as Select from '$lib/components/ui/select';
	import {
		LANGUAGES,
		type ModelKey,
		type SizeKey,
		findVariant
	} from '$lib/models';
	import { BACKENDS, type BackendPreference } from '$lib/webgpu';

	interface Props {
		selectedLanguage: ModelKey;
		selectedSize: SizeKey;
		backendPref: BackendPreference;
		disabled?: boolean;
		/** Additional grid/layout class on the outer section. */
		class?: string;
		onLanguageChange: (v: ModelKey) => void;
		onSizeChange: (v: SizeKey) => void;
		onBackendChange: (v: BackendPreference) => void;
	}

	let {
		selectedLanguage,
		selectedSize,
		backendPref,
		disabled = false,
		class: className = 'grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3',
		onLanguageChange,
		onSizeChange,
		onBackendChange
	}: Props = $props();

	const languagePack = $derived(LANGUAGES[selectedLanguage]);
	const modelSpec = $derived(findVariant(selectedLanguage, selectedSize));
	const backendSpec = $derived(BACKENDS[backendPref]);
</script>

<section class={className}>
	<label class="flex flex-col gap-1.5">
		<span class="text-muted-foreground text-xs font-medium tracking-wider uppercase">
			Language
		</span>
		<Select.Root
			type="single"
			value={selectedLanguage}
			onValueChange={(v) => v && onLanguageChange(v as ModelKey)}
			{disabled}
		>
			<Select.Trigger>
				{languagePack.label}
				<span class="text-muted-foreground ml-2 font-mono text-[10px]">
					· {languagePack.shortLabel}
				</span>
			</Select.Trigger>
			<Select.Content>
				{#each Object.values(LANGUAGES) as pack (pack.key)}
					<Select.Item value={pack.key} label={pack.label}>
						<span class="flex flex-col">
							<span>{pack.label}</span>
							<span class="text-muted-foreground text-[10px]">{pack.description}</span>
						</span>
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</label>

	<label class="flex flex-col gap-1.5">
		<span class="text-muted-foreground text-xs font-medium tracking-wider uppercase">
			Model size
		</span>
		<Select.Root
			type="single"
			value={selectedSize}
			onValueChange={(v) => v && onSizeChange(v as SizeKey)}
			{disabled}
		>
			<Select.Trigger>
				{modelSpec.sizeLabel}
				<span class="text-muted-foreground ml-2 font-mono text-[10px]">
					· {modelSpec.sizeNote}
				</span>
			</Select.Trigger>
			<Select.Content>
				{#each languagePack.variants as v (v.sizeKey)}
					<Select.Item value={v.sizeKey} label={v.sizeLabel}>
						<span class="flex flex-col">
							<span>
								{v.sizeLabel}
								{#if v.tagline}
									<span class="text-muted-foreground ml-1 text-[10px]">· {v.tagline}</span>
								{/if}
							</span>
							<span class="text-muted-foreground text-[10px] font-mono"
								>{v.sizeNote} · {v.modelId}</span
							>
						</span>
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</label>

	<label class="flex flex-col gap-1.5">
		<span class="text-muted-foreground text-xs font-medium tracking-wider uppercase">
			Inference
		</span>
		<Select.Root
			type="single"
			value={backendPref}
			onValueChange={(v) => v && onBackendChange(v as BackendPreference)}
			{disabled}
		>
			<Select.Trigger>{backendSpec.label}</Select.Trigger>
			<Select.Content>
				{#each Object.values(BACKENDS) as spec (spec.key)}
					<Select.Item value={spec.key} label={spec.label}>
						<span class="flex flex-col">
							<span>{spec.label}</span>
							<span class="text-muted-foreground text-[10px]">{spec.description}</span>
						</span>
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</label>
</section>
