<script lang="ts">
  import { thumbnailUrl, type Document } from '$lib/api';
  import { fmtTime } from '$lib/utils';
  import { cn } from '$lib/utils';

  type Props = {
    doc: Document;
    active?: boolean;
    onclick?: () => void;
  };
  let { doc, active = false, onclick }: Props = $props();
</script>

<button
  type="button"
  {onclick}
  aria-pressed={active}
  class={cn(
    'group flex flex-col overflow-hidden rounded-lg border bg-card text-left transition-all',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
    active
      ? 'border-primary ring-2 ring-primary shadow-md -translate-y-0.5'
      : 'border-border hover:border-primary hover:shadow-md hover:-translate-y-0.5',
  )}
>
  <div class="relative aspect-video w-full overflow-hidden bg-muted">
    <img
      src={thumbnailUrl(doc.doc_id)}
      alt=""
      loading="lazy"
      class="h-full w-full object-cover transition-transform group-hover:scale-105"
      onerror={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = 'hidden')}
    />
    {#if doc.duration}
      <span
        class="absolute bottom-1.5 right-1.5 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-white backdrop-blur-sm"
      >
        {fmtTime(doc.duration)}
      </span>
    {/if}
  </div>
  <div class="flex flex-1 flex-col gap-1 p-3">
    <div class="line-clamp-2 text-sm font-medium leading-snug">
      {doc.namn ?? doc.audio_path}
    </div>
    {#if doc.referenskod}
      <div class="font-mono text-[10px] text-muted-foreground">
        {doc.referenskod}
      </div>
    {/if}
  </div>
</button>
