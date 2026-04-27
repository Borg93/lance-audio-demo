<script lang="ts">
	import Shield from 'lucide-svelte/icons/shield-check';
	import Cpu from 'lucide-svelte/icons/cpu';
	import Package from 'lucide-svelte/icons/package';
	import Github from 'lucide-svelte/icons/github';
</script>

<main class="flex flex-1 justify-center p-4 md:p-6 lg:p-8">
	<article
		class="bg-card/95 relative w-full max-w-3xl overflow-hidden rounded-2xl border shadow-sm backdrop-blur-md"
	>
		<header class="border-b px-6 py-6 md:px-10">
			<div class="w-full space-y-1.5">
				<h1 class="text-2xl font-semibold tracking-tight md:text-3xl">About this demo</h1>
				<p class="text-muted-foreground text-sm">
					A local-first, multimodal playground for browser ML — no servers, no data upload.
				</p>
			</div>
		</header>

		<div class="flex w-full flex-col gap-6 px-6 py-8 md:px-10">
		<section class="bg-muted/40 flex items-start gap-4 rounded-xl border p-5">
			<Shield class="text-primary mt-0.5 size-5 shrink-0" />
			<div class="space-y-1 text-sm">
				<h2 class="text-base font-semibold">100% local</h2>
				<p class="text-muted-foreground leading-relaxed">
					Everything — audio, model weights, inference, and the resulting transcripts — stays in
					this browser tab. No microphone feed, tab audio, image, or text is ever uploaded. You
					can even disconnect from the internet after the model has loaded.
				</p>
			</div>
		</section>

		<section class="bg-muted/40 flex items-start gap-4 rounded-xl border p-5">
			<Cpu class="text-primary mt-0.5 size-5 shrink-0" />
			<div class="space-y-1 text-sm">
				<h2 class="text-base font-semibold">Runs on your own hardware</h2>
				<p class="text-muted-foreground leading-relaxed">
					Every model in this app runs on
					<span class="text-foreground font-medium">your machine</span> — your graphics card when
					one is available, falling back to your CPU when it isn't. The cost to you is a one-time
					download of the model weights; after that, inference happens at close to native speed,
					privately, and offline. No third-party server ever sees your audio, your transcripts,
					or how you use the tool.
				</p>
			</div>
		</section>

		<section class="bg-muted/40 flex items-start gap-4 rounded-xl border p-5">
			<Package class="text-primary mt-0.5 size-5 shrink-0" />
			<div class="space-y-1 text-sm">
				<h2 class="text-base font-semibold">Models in use</h2>
				<ul class="text-muted-foreground space-y-2 leading-relaxed">
					<li>
						<span class="text-foreground font-medium">Swedish:</span>
						<a
							href="https://huggingface.co/KBLab/kb-whisper-base"
							target="_blank"
							rel="noreferrer"
							class="text-foreground font-mono text-xs underline underline-offset-2"
							>KBLab/kb-whisper-base</a
						>
						— the National Library of Sweden's Swedish finetune of Whisper.
					</li>
					<li>
						<span class="text-foreground font-medium">English:</span>
						<a
							href="https://huggingface.co/onnx-community/whisper-base"
							target="_blank"
							rel="noreferrer"
							class="text-foreground font-mono text-xs underline underline-offset-2"
							>onnx-community/whisper-base</a
						>
						— the ONNX-exported, browser-optimised build of OpenAI's Whisper base model.
					</li>
				</ul>
			</div>
		</section>

		<section class="bg-muted/40 flex items-start gap-4 rounded-xl border p-5">
			<Github class="text-primary mt-0.5 size-5 shrink-0" />
			<div class="space-y-1 text-sm">
				<h2 class="text-base font-semibold">Stack</h2>
				<p class="text-muted-foreground leading-relaxed">
					SvelteKit 2 (Svelte 5 runes) · Tailwind v4 · shadcn-svelte · bits-ui ·
					<a
						href="https://bun.sh"
						target="_blank"
						rel="noreferrer"
						class="text-foreground underline underline-offset-2">Bun</a
					>
					· Transformers.js v4 · ONNX Runtime Web.
				</p>
			</div>
		</section>
		</div>
	</article>
</main>
