export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.DSZ7-eW4.js",app:"_app/immutable/entry/app.CphG8i7A.js",imports:["_app/immutable/entry/start.DSZ7-eW4.js","_app/immutable/chunks/CLd2CBe5.js","_app/immutable/chunks/BMgjO9vM.js","_app/immutable/chunks/C7JtvbPz.js","_app/immutable/entry/app.CphG8i7A.js","_app/immutable/chunks/C0vnYQH8.js","_app/immutable/chunks/BMgjO9vM.js","_app/immutable/chunks/CefDVh6V.js","_app/immutable/chunks/C7JtvbPz.js","_app/immutable/chunks/P2myRLKW.js","_app/immutable/chunks/CMHk5ZyV.js","_app/immutable/chunks/CX8ud67W.js","_app/immutable/chunks/DRk5mItz.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js')),
			__memo(() => import('./nodes/3.js')),
			__memo(() => import('./nodes/4.js')),
			__memo(() => import('./nodes/5.js')),
			__memo(() => import('./nodes/6.js')),
			__memo(() => import('./nodes/7.js')),
			__memo(() => import('./nodes/8.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			},
			{
				id: "/annotate",
				pattern: /^\/annotate\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 6 },
				endpoint: null
			},
			{
				id: "/atlas",
				pattern: /^\/atlas\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 3 },
				endpoint: null
			},
			{
				id: "/diagram",
				pattern: /^\/diagram\/?$/,
				params: [],
				page: null,
				endpoint: __memo(() => import('./entries/endpoints/diagram/_server.ts.js'))
			},
			{
				id: "/graph",
				pattern: /^\/graph\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 8 },
				endpoint: null
			},
			{
				id: "/guide",
				pattern: /^\/guide\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 4 },
				endpoint: null
			},
			{
				id: "/tree",
				pattern: /^\/tree\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 7 },
				endpoint: null
			},
			{
				id: "/workflow",
				pattern: /^\/workflow\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 5 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
