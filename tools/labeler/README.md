# labeler (throwaway)

Standalone Bun tool for manually relabeling mis-detected language directories under `input/<lang>/`. Not part of the main app — feel free to delete `tools/labeler/` when you're done.

## Run

```bash
cd tools/labeler
bun run labeler.ts --root ../../input --port 3999
# open http://localhost:3999
```

## Use

1. Home page lists every `input/<lang>/` subdir with file counts.
2. Click a language to drill in.
3. Per file, video auto-plays; use:
   - `1` → move to `input/sv/`
   - `2` → move to `input/en/`
   - `3` → skip (don't move, advance)
   - `o` → prompt for any other 2–3 letter code
   - `n` / `p` → next / prev
4. Files are moved on disk immediately (atomic `rename`).

## Safety

- Only MP4s are listed/served/moved.
- Paths are confined to `--root` (no escaping the input tree).
- There's no database, no Lance, no coupling to the main app.
