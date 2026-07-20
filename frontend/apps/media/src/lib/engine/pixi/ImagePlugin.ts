import { Application, ColorMatrixFilter, Graphics, Sprite, Texture } from "pixi.js";
import type { ViewportBounds } from "./types.js";

export class ImagePlugin {
  private app: Application;
  private sprite: Sprite | null = null;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  /** Pointer id captured for the active pan drag, or null when not panning */
  private panPointerId: number | null = null;

  zoom = 1;
  panX = 0;
  panY = 0;
  imageWidth = 0;
  imageHeight = 0;
  /** The zoom level at which the image fits the viewport (baseline = 100%) */
  private fitScale = 1;
  /** Reused across adjustments so we don't leak a filter per call */
  private colorFilter: ColorMatrixFilter | null = null;

  onViewportChange?: (bounds: ViewportBounds) => void;
  canPan?: () => boolean;

  constructor(app: Application) {
    this.app = app;
    // GPU-accelerated transforms for the viewport container
    app.stage.isRenderGroup = true;
    this.setupEvents();
  }

  async load(url: string): Promise<void> {
    if (this.sprite) {
      this.app.stage.removeChild(this.sprite);
      // Destroy the texture/source we are replacing to free GPU memory
      const old = this.sprite.texture;
      this.sprite.destroy();
      old.destroy(true);
    }

    const img = new Image();
    img.crossOrigin = "anonymous";
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
      img.src = url;
    });

    const texture = Texture.from(img);
    this.sprite = new Sprite(texture);
    this.imageWidth = img.naturalWidth;
    this.imageHeight = img.naturalHeight;

    // Add as bottom layer on stage
    this.app.stage.addChildAt(this.sprite, 0);
    this.fitToViewport();
  }

  /**
   * Load image from Arrow Binary column data (zero-copy path).
   * The bytes go from Arrow buffer → Blob → ObjectURL → Image → GPU texture.
   * No extra HTTP request, no separate endpoint.
   */
  async loadFromBytes(bytes: Uint8Array, mimeType = "image/png"): Promise<void> {
    if (this.sprite) {
      this.app.stage.removeChild(this.sprite);
      // Destroy the texture/source we are replacing to free GPU memory
      const old = this.sprite.texture;
      this.sprite.destroy();
      old.destroy(true);
    }

    // Ensure we have a proper ArrayBuffer-backed view for the Blob constructor
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const blob = new Blob([buffer as ArrayBuffer], { type: mimeType });
    const url = URL.createObjectURL(blob);

    try {
      const img = new Image();
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error("Failed to load image from Arrow binary data"));
        img.src = url;
      });

      const texture = Texture.from(img);
      this.sprite = new Sprite(texture);
      this.imageWidth = img.naturalWidth;
      this.imageHeight = img.naturalHeight;

      this.app.stage.addChildAt(this.sprite, 0);
      this.fitToViewport();
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  fitToViewport(): void {
    if (!this.sprite) return;

    const vw = this.app.screen.width;
    const vh = this.app.screen.height;
    const scale = Math.min(vw / this.imageWidth, vh / this.imageHeight);

    this.fitScale = scale;
    this.zoom = scale;
    this.panX = (vw - this.imageWidth * scale) / 2;
    this.panY = (vh - this.imageHeight * scale) / 2;

    this.applyTransform();
  }

  getViewportBounds(): ViewportBounds {
    const vw = this.app.screen.width;
    const vh = this.app.screen.height;
    return {
      x: -this.panX / this.zoom,
      y: -this.panY / this.zoom,
      width: vw / this.zoom,
      height: vh / this.zoom,
    };
  }

  destroy(): void {
    this.removeEvents();
    if (this.sprite) {
      // Tear down the sprite together with its texture/source to free GPU memory
      this.sprite.destroy({ texture: true, textureSource: true });
      this.sprite = null;
    }
  }

  private applyTransform(): void {
    this.app.stage.scale.set(this.zoom);
    this.app.stage.position.set(this.panX, this.panY);
    this.onViewportChange?.(this.getViewportBounds());
    // Render on demand — no continuous ticker
    this.app.render();
  }

  private get canvas(): HTMLCanvasElement {
    return this.app.canvas as HTMLCanvasElement;
  }

  private setupEvents(): void {
    this.canvas.addEventListener("wheel", this.onWheel, { passive: false });
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    this.canvas.addEventListener("pointermove", this.onPointerMove);
    this.canvas.addEventListener("pointerup", this.onPointerUp);
    // No pointerleave→up shortcut: pointer capture keeps the pan alive
    // even when the cursor leaves the canvas mid-gesture.
    this.canvas.addEventListener("dblclick", this.onDoubleClick);
  }

  private removeEvents(): void {
    this.canvas.removeEventListener("wheel", this.onWheel);
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerup", this.onPointerUp);
    this.canvas.removeEventListener("dblclick", this.onDoubleClick);
  }

  private zoomAt(x: number, y: number, factor: number): void {
    const newZoom = Math.max(0.05, Math.min(50, this.zoom * factor));
    this.panX = x - (x - this.panX) * (newZoom / this.zoom);
    this.panY = y - (y - this.panY) * (newZoom / this.zoom);
    this.zoom = newZoom;
  }

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();

    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (e.ctrlKey) {
      // Pinch-to-zoom (trackpad): deltaY is continuous, clamp and scale gently
      const delta = Math.max(-20, Math.min(20, e.deltaY));
      this.zoomAt(mx, my, 1 - delta * 0.003);
    } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY) * 0.5) {
      // Trackpad two-finger pan (deltaX is significant)
      this.panX -= e.deltaX;
      this.panY -= e.deltaY;
    } else {
      // Mouse wheel zoom: use deltaY magnitude for proportional zoom
      // Normalize: most mice send deltaY ~100 per tick, trackpads send ~1-10
      const normalized = Math.max(-150, Math.min(150, e.deltaY));
      const factor = 1 - normalized * 0.001;
      this.zoomAt(mx, my, factor);
    }

    this.applyTransform();
  };

  private onPointerDown = (e: PointerEvent): void => {
    if (this.canPan && !this.canPan()) return;
    this.dragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.canvas.style.cursor = "grabbing";
    // Capture so the pan survives the cursor leaving the canvas mid-drag
    this.panPointerId = e.pointerId;
    this.canvas.setPointerCapture(e.pointerId);
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.dragging) return;

    this.panX += e.clientX - this.lastX;
    this.panY += e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;

    this.applyTransform();
  };

  private onPointerUp = (e: PointerEvent): void => {
    if (this.dragging) {
      this.dragging = false;
      this.canvas.style.cursor = "default";
    }
    if (this.panPointerId !== null) {
      this.canvas.releasePointerCapture(this.panPointerId);
      this.panPointerId = null;
    }
  };

  private onDoubleClick = (): void => {
    // Double-click intentionally does nothing — use the zoom controls to reset view
  };

  /** Apply image adjustments via ColorMatrixFilter */
  setImageAdjustments(brightness: number, contrast: number, saturation: number): void {
    if (!this.sprite) return;

    // All at defaults (1.0) — remove filter for zero overhead
    // (keep the cached instance around for reuse on the next non-default call)
    if (brightness === 1 && contrast === 1 && saturation === 1) {
      this.sprite.filters = [];
      this.app.render();
      return;
    }

    // Lazily create one filter and reconfigure it in place on every call
    const filter = (this.colorFilter ??= new ColorMatrixFilter());
    filter.reset();
    filter.brightness(brightness, false);
    filter.contrast(contrast - 1, true); // pixi contrast: 0 = normal, -1 = grey, +1 = high
    filter.saturate(saturation - 1, true); // pixi saturate: 0 = normal, -1 = desaturated
    this.sprite.filters = [filter];
    this.app.render();
  }

  /** Zoom percentage relative to fit-to-viewport (fit = 100%) */
  get zoomPercent(): number {
    return this.fitScale > 0 ? this.zoom / this.fitScale : 1;
  }

  /** Programmatic reset — callable from UI regardless of tool */
  resetView(): void {
    this.fitToViewport();
  }

  /** Zoom in by a fixed step, centered on canvas */
  zoomIn(): void {
    const vw = this.app.screen.width;
    const vh = this.app.screen.height;
    this.zoomAt(vw / 2, vh / 2, 1.25);
    this.applyTransform();
  }

  /** Zoom out by a fixed step, centered on canvas */
  zoomOut(): void {
    const vw = this.app.screen.width;
    const vh = this.app.screen.height;
    this.zoomAt(vw / 2, vh / 2, 0.8);
    this.applyTransform();
  }
}
