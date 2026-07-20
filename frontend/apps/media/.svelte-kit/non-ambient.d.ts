
// this file is generated — do not edit it


declare module "svelte/elements" {
	export interface HTMLAttributes<T> {
		'data-sveltekit-keepfocus'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-noscroll'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-preload-code'?:
			| true
			| ''
			| 'eager'
			| 'viewport'
			| 'hover'
			| 'tap'
			| 'off'
			| undefined
			| null;
		'data-sveltekit-preload-data'?: true | '' | 'hover' | 'tap' | 'off' | undefined | null;
		'data-sveltekit-reload'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-replacestate'?: true | '' | 'off' | undefined | null;
	}
}

export {};


declare module "$app/types" {
	type MatcherParam<M> = M extends (param : string) => param is (infer U extends string) ? U : string;

	export interface AppTypes {
		RouteId(): "/" | "/annotate" | "/atlas" | "/diagram" | "/graph" | "/guide" | "/tree" | "/workflow";
		RouteParams(): {
			
		};
		LayoutParams(): {
			"/": Record<string, never>;
			"/annotate": Record<string, never>;
			"/atlas": Record<string, never>;
			"/diagram": Record<string, never>;
			"/graph": Record<string, never>;
			"/guide": Record<string, never>;
			"/tree": Record<string, never>;
			"/workflow": Record<string, never>
		};
		Pathname(): "/" | "/annotate" | "/atlas" | "/diagram" | "/graph" | "/guide" | "/tree" | "/workflow";
		ResolvedPathname(): `${"" | `/${string}`}${ReturnType<AppTypes['Pathname']>}`;
		Asset(): string & {};
	}
}