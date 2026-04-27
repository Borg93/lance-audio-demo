<script lang="ts">
	import type { HTMLButtonAttributes } from 'svelte/elements';
	import { cn } from '$lib/utils';

	type Props = HTMLButtonAttributes & {
		shimmerColor?: string;
		shimmerSize?: string;
		borderRadius?: string;
		shimmerDuration?: string;
		background?: string;
		class?: string;
	};

	let {
		class: className,
		shimmerColor = '#ffffff',
		shimmerSize = '0.05em',
		borderRadius = '100px',
		shimmerDuration = '3s',
		background = 'rgba(0, 0, 0, 1)',
		children,
		...rest
	}: Props = $props();
</script>

<button
	style:--spread="90deg"
	style:--shimmer-color={shimmerColor}
	style:--radius={borderRadius}
	style:--speed={shimmerDuration}
	style:--cut={shimmerSize}
	style:--bg={background}
	class={cn(
		'group relative z-0 inline-flex cursor-pointer items-center justify-center overflow-hidden px-6 py-3 text-sm font-medium whitespace-nowrap text-white transition-transform active:translate-y-px',
		'disabled:pointer-events-none disabled:opacity-50',
		'[background:var(--bg)] [border-radius:var(--radius)] [transform:translateZ(0)]',
		className
	)}
	{...rest}
>
	<!-- shimmer slide track -->
	<div
		class="absolute inset-0 overflow-visible blur-[2px] [container-type:size]"
		aria-hidden="true"
	>
		<div
			class="animate-shimmer-slide absolute inset-0 h-[100cqh] w-[100cqh] [aspect-ratio:1]"
		>
			<div
				class="animate-spin-around absolute -inset-full w-auto rotate-0 [background:conic-gradient(from_calc(270deg-(var(--spread)*0.5)),transparent_0,var(--shimmer-color)_var(--spread),transparent_var(--spread))] [translate:none]"
			></div>
		</div>
	</div>

	<!-- inner cut so only the outer ring shimmers -->
	<div
		class="absolute [background:var(--bg)] [border-radius:calc(var(--radius)-var(--cut))] [inset:var(--cut)]"
		aria-hidden="true"
	></div>

	<!-- subtle inset highlight -->
	<div
		class="absolute inset-0 size-full rounded-[inherit] shadow-[inset_0_-8px_10px_#ffffff1f] transition-all duration-300 ease-in-out group-hover:shadow-[inset_0_-6px_10px_#ffffff3f] group-active:shadow-[inset_0_-10px_10px_#ffffff3f]"
		aria-hidden="true"
	></div>

	<span class="relative z-10 flex items-center justify-center gap-2">
		{@render children?.()}
	</span>
</button>
