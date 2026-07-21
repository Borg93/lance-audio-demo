/**
 * Shared OpenCV.js loader for the CV tools (currently: magnetic corner-snap).
 *
 * `@techstark/opencv-js` is a CJS module (`main: dist/opencv.js`) whose exports are
 * attached at RUNTIME — so under a bundler, `import * as ns` yields a namespace whose
 * `ns.Mat` is undefined and whose properties are sealed (an `ns.onRuntimeInitialized =`
 * assignment silently no-ops). The old in-tool loaders did exactly that and hung forever.
 * This helper unwraps the interop default export and handles both ready styles (an
 * `onRuntimeInitialized` callback, or builds whose default export is a ready-promise).
 */

export type CV = typeof import("@techstark/opencv-js");

let loaded: Promise<CV> | null = null;

export function loadOpenCV(): Promise<CV> {
  // One in-flight/shared load — the 8MB wasm is paid once, both tools reuse it.
  loaded ??= (async () => {
    const mod: Record<string, unknown> = await import("@techstark/opencv-js");
    let cv = (mod.default ?? mod) as Record<string, unknown>;
    if (typeof (cv as { then?: unknown }).then === "function") {
      // Some builds default-export a promise that resolves to the ready cv object.
      cv = (await cv) as Record<string, unknown>;
    } else if (!cv.Mat) {
      // Wait for the wasm runtime. The callback alone is RACY — if the runtime finished
      // initializing between our check and the assignment, it never fires — so poll too.
      await new Promise<void>((resolve) => {
        const iv = setInterval(() => {
          if (cv.Mat) {
            clearInterval(iv);
            resolve();
          }
        }, 25);
        (cv as { onRuntimeInitialized?: () => void }).onRuntimeInitialized = () => {
          clearInterval(iv);
          resolve();
        };
      });
    }
    return cv as unknown as CV;
  })();
  return loaded;
}
