<script lang="ts">
  /**
   * gpu-scatter — a self-contained WebGPU point-scatter renderer for the atlas.
   *
   * Modern WebGPU only: the point renderer uses instanced quads (WebGPU has no
   * point-size primitive) shaded into round, premultiplied-alpha sprites. The
   * component owns ONLY pixels + camera + pointer gestures; all data/colour/
   * selection logic stays in AtlasMap, which feeds us premultiply-free RGBA
   * bytes and receives DATA-space callbacks.
   *
   *   • onHover(dataX, dataY)  — rAF-throttled cursor → data coords
   *   • onHoverEnd()           — pointer left the canvas
   *   • onPick(dataX, dataY)   — a click (drag-free) at data coords
   *   • onLasso(flatDataXY)    — lasso polygon vertices in DATA coords
   *
   * Camera is data-space pan/zoom (cursor-anchored wheel). The lasso draws on a
   * separate 2D overlay so the GPU surface is never disturbed mid-gesture.
   */

  type Mode = 'lasso' | 'pan';

  let {
    x,
    y,
    rgba,
    pointSize,
    width,
    height,
    background,
    onHover,
    onHoverEnd,
    onPick,
    onLasso,
    mode,
    markerXY = null,
  }: {
    x: Float32Array;
    y: Float32Array;
    rgba: Uint8Array; // length 4*count, RGBA bytes; shader premultiplies
    pointSize: number; // device px diameter; clamped to a sane minimum
    width: number; // css px
    height: number; // css px
    background: string; // css hex, e.g. '#0a0a0a'
    onHover: (dataX: number, dataY: number) => void;
    onHoverEnd: () => void;
    onPick: (dataX: number, dataY: number) => void;
    onLasso: (polyDataXY: number[]) => void; // flat [x0,y0,x1,y1,...] in DATA coords
    mode: Mode;
    markerXY?: [number, number] | null; // active point in DATA coords → highlight ring
  } = $props();

  const dpr = Math.min(globalThis.devicePixelRatio ?? 1, 1.5);

  let gpuCanvas = $state<HTMLCanvasElement | null>(null);
  let overlayCanvas = $state<HTMLCanvasElement | null>(null);
  let gpuError = $state<string | null>(null);

  // ── camera (data coords at canvas centre + device-px per data unit) ───────
  let center = $state.raw<[number, number]>([0, 0]);
  let scale = $state(1);

  // ── GPU handles (plain refs; not reactive) ────────────────────────────────
  let device: GPUDevice | null = null;
  let ctx: GPUCanvasContext | null = null;
  let pipeline: GPURenderPipeline | null = null;
  let uniformBuffer: GPUBuffer | null = null;
  let uniformBindGroup: GPUBindGroup | null = null;
  let posBuffer: GPUBuffer | null = null;
  let colorBuffer: GPUBuffer | null = null;
  let count = 0;
  let ready = false; // device + pipeline initialised

  // identity tracking so we re-upload only when a prop's buffer actually swaps
  let lastXY: Float32Array | null = null;
  let lastRgba: Uint8Array | null = null;

  // ── clear colour parsed from the background hex (0..1, opaque) ────────────
  function clearRgb(hex: string): GPUColor {
    let h = hex.trim().replace('#', '');
    if (h.length === 3) h = h[0]! + h[0]! + h[1]! + h[1]! + h[2]! + h[2]!;
    const n = Number.parseInt(h, 16);
    if (!Number.isFinite(n)) return { r: 0, g: 0, b: 0, a: 1 };
    return { r: ((n >> 16) & 255) / 255, g: ((n >> 8) & 255) / 255, b: (n & 255) / 255, a: 1 };
  }

  const WGSL = `
struct U {
  center : vec2f,   // data coords shown at canvas centre
  scale  : f32,     // device px per data unit
  pointSize : f32,  // point diameter, device px
  resolution : vec2f, // device px
  _pad : vec2f,
};
@group(0) @binding(0) var<uniform> u : U;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) color : vec4f,
  @location(1) quad  : vec2f,   // [-0.5,0.5] local, for the round mask
};

@vertex
fn vs(@builtin(vertex_index) vi : u32,
      @location(0) iPos : vec2f,
      @location(1) iColor : vec4f) -> VSOut {
  var C = array<vec2f,6>(
    vec2f(-0.5,-0.5), vec2f(0.5,-0.5), vec2f(-0.5,0.5),
    vec2f(-0.5,0.5),  vec2f(0.5,-0.5), vec2f(0.5,0.5));
  let corner = C[vi];
  let screen = (iPos - u.center) * u.scale + u.resolution * 0.5;
  let p = screen + corner * u.pointSize;
  let clip = (p / u.resolution) * 2.0 - vec2f(1.0, 1.0);
  var o : VSOut;
  o.pos = vec4f(clip.x, -clip.y, 0.0, 1.0);  // flip Y (screen is y-down)
  o.color = iColor;
  o.quad = corner;
  return o;
}

@fragment
fn fs(@location(0) color : vec4f, @location(1) quad : vec2f) -> @location(0) vec4f {
  if (color.a < 0.02) { discard; }          // hidden points
  if (dot(quad, quad) > 0.25) { discard; }  // round point
  return vec4f(color.rgb * color.a, color.a); // premultiplied
}
`;

  async function initGpu(canvas: HTMLCanvasElement): Promise<boolean> {
    if (!navigator.gpu) {
      gpuError = 'WebGPU is required (not available in this browser)';
      return false;
    }
    let adapter: GPUAdapter | null = null;
    try {
      adapter = await navigator.gpu.requestAdapter();
    } catch {
      adapter = null;
    }
    if (!adapter) {
      gpuError = 'WebGPU is required (not available in this browser)';
      return false;
    }
    let dev: GPUDevice;
    try {
      dev = await adapter.requestDevice();
    } catch {
      gpuError = 'WebGPU is required (not available in this browser)';
      return false;
    }
    const context = canvas.getContext('webgpu');
    if (!context) {
      gpuError = 'WebGPU is required (not available in this browser)';
      return false;
    }
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: dev, format: fmt, alphaMode: 'premultiplied' });

    const module = dev.createShaderModule({ code: WGSL });
    const pl = dev.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: 8,
            stepMode: 'instance',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
          },
          {
            arrayStride: 4,
            stepMode: 'instance',
            attributes: [{ shaderLocation: 1, offset: 0, format: 'unorm8x4' }],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: 'fs',
        targets: [
          {
            format: fmt,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });

    const ubo = dev.createBuffer({
      size: 32, // center vec2f, scale f32, pointSize f32, resolution vec2f, pad vec2f
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = dev.createBindGroup({
      layout: pl.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: ubo } }],
    });

    device = dev;
    ctx = context;
    pipeline = pl;
    uniformBuffer = ubo;
    uniformBindGroup = bindGroup;
    ready = true;
    return true;
  }

  // ── data → GPU buffers ────────────────────────────────────────────────────
  function uploadPositions(): void {
    const dev = device;
    if (!dev) return;
    const n = Math.min(x.length, y.length);
    count = n;
    const interleaved = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      interleaved[i * 2] = x[i]!;
      interleaved[i * 2 + 1] = y[i]!;
    }
    if (posBuffer) posBuffer.destroy();
    const byteLength = Math.max(interleaved.byteLength, 4); // GPU buffers must be non-empty
    const buf = dev.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(buf, 0, interleaved.buffer, interleaved.byteOffset, interleaved.byteLength);
    posBuffer = buf;
  }

  function uploadColors(): void {
    const dev = device;
    if (!dev) return;
    if (colorBuffer) colorBuffer.destroy();
    const byteLength = Math.max(rgba.byteLength, 4); // GPU buffers must be non-empty
    const buf = dev.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(buf, 0, rgba.buffer, rgba.byteOffset, rgba.byteLength);
    colorBuffer = buf;
  }

  /** Fit the camera so the whole point cloud is visible & centred. */
  function fit(): void {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const n = Math.min(x.length, y.length);
    for (let i = 0; i < n; i++) {
      const xi = x[i]!;
      const yi = y[i]!;
      if (xi < minX) minX = xi;
      if (xi > maxX) maxX = xi;
      if (yi < minY) minY = yi;
      if (yi > maxY) maxY = yi;
    }
    if (!Number.isFinite(minX)) {
      minX = 0;
      minY = 0;
      maxX = 1;
      maxY = 1;
    }
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const w = Math.max(1, width);
    const h = Math.max(1, height);
    scale = 0.9 * Math.min((w * dpr) / spanX, (h * dpr) / spanY);
    center = [(minX + maxX) / 2, (minY + maxY) / 2];
  }

  // ── coalesced redraw (one rAF, never a synchronous loop) ──────────────────
  let drawRaf = 0;
  function requestDraw(): void {
    if (drawRaf !== 0) return;
    drawRaf = requestAnimationFrame(() => {
      drawRaf = 0;
      draw();
    });
  }

  function draw(): void {
    const dev = device;
    const context = ctx;
    const canvas = gpuCanvas;
    if (!dev || !context || !canvas || !pipeline || !uniformBuffer || !uniformBindGroup) return;
    const resW = Math.max(1, Math.round(width * dpr));
    const resH = Math.max(1, Math.round(height * dpr));
    if (canvas.width !== resW) canvas.width = resW;
    if (canvas.height !== resH) canvas.height = resH;

    // uniforms: center vec2f, scale f32, pointSize f32, resolution vec2f, pad vec2f
    const u = new Float32Array(8);
    u[0] = center[0];
    u[1] = center[1];
    u[2] = scale;
    u[3] = Math.max(1.5, pointSize);
    u[4] = resW;
    u[5] = resH;
    u[6] = 0;
    u[7] = 0;
    dev.queue.writeBuffer(uniformBuffer, 0, u.buffer, u.byteOffset, u.byteLength);

    const enc = dev.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: clearRgb(background),
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    if (count > 0 && posBuffer && colorBuffer) {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, uniformBindGroup);
      pass.setVertexBuffer(0, posBuffer);
      pass.setVertexBuffer(1, colorBuffer);
      pass.draw(6, count);
    }
    pass.end();
    dev.queue.submit([enc.finish()]);
  }

  // ── coord helpers ─────────────────────────────────────────────────────────
  /** CSS-px (relative to canvas) → data coords. */
  function screenToData(cssX: number, cssY: number): [number, number] {
    const resW = width * dpr;
    const resH = height * dpr;
    const dx = (cssX * dpr - resW * 0.5) / scale + center[0];
    const dy = (cssY * dpr - resH * 0.5) / scale + center[1];
    return [dx, dy];
  }

  // ── lifecycle: init once the canvas mounts ────────────────────────────────
  $effect(() => {
    const canvas = gpuCanvas;
    if (!canvas) return;
    let cancelled = false;
    (async () => {
      const ok = await initGpu(canvas);
      if (cancelled || !ok) return;
      uploadPositions();
      uploadColors();
      lastXY = x;
      lastRgba = rgba;
      fit();
      requestDraw();
    })();
    return () => {
      cancelled = true;
      if (drawRaf !== 0) {
        cancelAnimationFrame(drawRaf);
        drawRaf = 0;
      }
      if (posBuffer) posBuffer.destroy();
      if (colorBuffer) colorBuffer.destroy();
      if (uniformBuffer) uniformBuffer.destroy();
      if (device) device.destroy();
      device = null;
      ctx = null;
      pipeline = null;
      uniformBuffer = null;
      uniformBindGroup = null;
      posBuffer = null;
      colorBuffer = null;
      ready = false;
    };
  });

  // Re-upload + refit when x/y swap identity (a space change / new data).
  $effect(() => {
    void x;
    void y;
    if (!ready || x === lastXY) return;
    uploadPositions();
    lastXY = x;
    fit();
    requestDraw();
  });

  // Re-upload colours when the rgba buffer swaps identity (recolour / filter).
  $effect(() => {
    void rgba;
    if (!ready || rgba === lastRgba) return;
    uploadColors();
    lastRgba = rgba;
    requestDraw();
  });

  // Redraw on any camera / size / point-size change.
  $effect(() => {
    void center;
    void scale;
    void width;
    void height;
    void pointSize;
    void background;
    if (ready) requestDraw();
  });

  // ── pointer gestures ──────────────────────────────────────────────────────
  type Drag = {
    kind: 'pan' | 'lasso';
    startX: number; // css px
    startY: number;
    lastX: number;
    lastY: number;
    moved: boolean;
    lassoPts: [number, number][]; // css px vertices
  };
  let drag: Drag | null = null;
  const CLICK_SLOP = 4; // css px; below this a drag counts as a click

  // rAF-throttled hover (never write reactive state per raw pointer event)
  let hoverRaf = 0;
  let pendingHover: [number, number] | null = null;
  function flushHover(): void {
    hoverRaf = 0;
    if (pendingHover) onHover(pendingHover[0], pendingHover[1]);
  }

  function localCss(e: PointerEvent | WheelEvent): [number, number] {
    const rect = gpuCanvas?.getBoundingClientRect();
    if (!rect) return [0, 0];
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  function onPointerDown(e: PointerEvent): void {
    const [cx, cy] = localCss(e);
    gpuCanvas?.setPointerCapture(e.pointerId);
    // Pan when: middle button, OR shift+left, OR pan-mode left.
    const wantPan = e.button === 1 || (e.button === 0 && (e.shiftKey || mode === 'pan'));
    drag = {
      kind: wantPan ? 'pan' : 'lasso',
      startX: cx,
      startY: cy,
      lastX: cx,
      lastY: cy,
      moved: false,
      lassoPts: wantPan ? [] : [[cx, cy]],
    };
  }

  function onPointerMove(e: PointerEvent): void {
    const [cx, cy] = localCss(e);
    if (drag) {
      const dxCss = cx - drag.lastX;
      const dyCss = cy - drag.lastY;
      drag.lastX = cx;
      drag.lastY = cy;
      if (Math.abs(cx - drag.startX) > CLICK_SLOP || Math.abs(cy - drag.startY) > CLICK_SLOP) {
        drag.moved = true;
      }
      if (drag.kind === 'pan') {
        // center -= deltaDevicePx / scale
        center = [center[0] - (dxCss * dpr) / scale, center[1] - (dyCss * dpr) / scale];
      } else {
        drag.lassoPts.push([cx, cy]);
        drawLassoOverlay();
      }
      return;
    }
    // plain hover (rAF-coalesced)
    const [dx, dy] = screenToData(cx, cy);
    pendingHover = [dx, dy];
    if (hoverRaf === 0) hoverRaf = requestAnimationFrame(flushHover);
  }

  function onPointerUp(e: PointerEvent): void {
    const d = drag;
    drag = null;
    if (gpuCanvas?.hasPointerCapture(e.pointerId)) gpuCanvas.releasePointerCapture(e.pointerId);
    clearLassoOverlay();
    if (!d) return;
    const [cx, cy] = localCss(e);
    if (!d.moved) {
      // a click → pick at data coords
      const [dx, dy] = screenToData(cx, cy);
      onPick(dx, dy);
      return;
    }
    if (d.kind === 'lasso' && d.lassoPts.length >= 3) {
      const flat: number[] = [];
      for (const [px, py] of d.lassoPts) {
        const [dx, dy] = screenToData(px, py);
        flat.push(dx, dy);
      }
      onLasso(flat);
    }
  }

  function onPointerLeave(): void {
    if (hoverRaf !== 0) {
      cancelAnimationFrame(hoverRaf);
      hoverRaf = 0;
    }
    pendingHover = null;
    onHoverEnd();
  }

  function onWheel(e: WheelEvent): void {
    e.preventDefault();
    const [cx, cy] = localCss(e);
    // data point under the cursor before zoom
    const [dx, dy] = screenToData(cx, cy);
    const next = Math.max(1e-4, Math.min(1e7, scale * Math.exp(-e.deltaY * 0.001)));
    // adjust center so (dx,dy) stays under the cursor after the scale change
    const resW = width * dpr;
    const resH = height * dpr;
    const cxDev = cx * dpr - resW * 0.5;
    const cyDev = cy * dpr - resH * 0.5;
    scale = next;
    center = [dx - cxDev / next, dy - cyDev / next];
  }

  // ── lasso overlay (2D canvas, kept off the GPU surface) ───────────────────
  function drawLassoOverlay(): void {
    const c = overlayCanvas;
    const d = drag;
    if (!c || !d || d.kind !== 'lasso') return;
    if (c.width !== Math.round(width * dpr)) c.width = Math.round(width * dpr);
    if (c.height !== Math.round(height * dpr)) c.height = Math.round(height * dpr);
    const ctx2d = c.getContext('2d');
    if (!ctx2d) return;
    ctx2d.clearRect(0, 0, c.width, c.height);
    if (d.lassoPts.length < 2) return;
    ctx2d.save();
    ctx2d.scale(dpr, dpr);
    ctx2d.lineWidth = 1.5;
    ctx2d.strokeStyle = 'rgba(99,102,241,0.9)';
    ctx2d.fillStyle = 'rgba(99,102,241,0.12)';
    ctx2d.beginPath();
    const first = d.lassoPts[0]!;
    ctx2d.moveTo(first[0], first[1]);
    for (let i = 1; i < d.lassoPts.length; i++) ctx2d.lineTo(d.lassoPts[i]![0], d.lassoPts[i]![1]);
    ctx2d.closePath();
    ctx2d.fill();
    ctx2d.stroke();
    ctx2d.restore();
  }

  function clearLassoOverlay(): void {
    const c = overlayCanvas;
    if (!c) return;
    const ctx2d = c.getContext('2d');
    ctx2d?.clearRect(0, 0, c.width, c.height);
  }

  $effect(() => () => {
    if (hoverRaf !== 0) cancelAnimationFrame(hoverRaf);
  });

  const cursor = $derived(mode === 'pan' ? 'grab' : 'crosshair');

  // Screen position (css px) of the active-point highlight ring, recomputed as
  // the camera pans/zooms (reads center/scale — the same transform `screenToData`
  // inverts). Hidden when the point is off-screen.
  const markerScreen = $derived.by((): { left: number; top: number } | null => {
    if (!markerXY) return null;
    const left = ((markerXY[0] - center[0]) * scale) / dpr + width / 2;
    const top = ((markerXY[1] - center[1]) * scale) / dpr + height / 2;
    if (left < -8 || left > width + 8 || top < -8 || top > height + 8) return null;
    return { left, top };
  });
</script>

{#if gpuError}
  <div class="grid h-full place-items-center p-6 text-center text-sm text-destructive">
    {gpuError} — the embedding map needs WebGPU.
  </div>
{:else}
  <canvas
    bind:this={gpuCanvas}
    class="absolute inset-0 block touch-none"
    style:width="{width}px"
    style:height="{height}px"
    style:cursor={cursor}
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    onpointerleave={onPointerLeave}
    onwheel={onWheel}
  ></canvas>
  <canvas
    bind:this={overlayCanvas}
    class="pointer-events-none absolute inset-0 block"
    style:width="{width}px"
    style:height="{height}px"
  ></canvas>
  <!-- subtle dot grid backing the embedding space (theme-neutral grey) -->
  <div
    class="pointer-events-none absolute inset-0"
    style:background-image="radial-gradient(rgba(130,130,130,0.22) 1.1px, transparent 1.6px)"
    style:background-size="22px 22px"
  ></div>
  {#if markerScreen}
    <!-- highlight ring for the active hit (the point shown in the player / table) -->
    <div
      class="pointer-events-none absolute z-10 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary"
      style:left="{markerScreen.left}px"
      style:top="{markerScreen.top}px"
      style:box-shadow="0 0 0 2px rgba(0,0,0,0.45), 0 0 10px 2px var(--color-primary)"
    ></div>
  {/if}
{/if}
