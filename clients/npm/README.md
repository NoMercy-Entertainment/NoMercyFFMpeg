# @nomercy-entertainment/ffmpeg-static

Fetches the **NoMercy FFmpeg fork** binary for the current platform, on first use.

This is the fork the NoMercy MediaServer ships: custom muxers, AACS/BD+ decrypt,
libass subtitle burn-in, and the codec set the product relies on. It is **not
stock ffmpeg** — fixtures, transcodes, and probes produced with it match what
the product produces in production. One package, so every CI action and every
NoMercy library that needs ffmpeg resolves the same binary the same way.

Binaries come from the [`nomercy-ffmpeg`](https://github.com/NoMercy-Entertainment/nomercy-ffmpeg)
GitHub releases. The package downloads the matching per-platform artifact,
extracts it, caches it under the OS temp dir, and returns an absolute path.

## Install

```sh
npm install @nomercy-entertainment/ffmpeg-static
```

## Programmatic use

```ts
import { ensureFfmpeg, ensureFfprobe } from '@nomercy-entertainment/ffmpeg-static';

const ffmpeg = await ensureFfmpeg(); // absolute path, downloaded on first call
const ffprobe = await ensureFfprobe();

// then spawn it
import { spawnSync } from 'node:child_process';
spawnSync(ffmpeg, ['-version'], { stdio: 'inherit' });
```

`ffmpegPath()` and `ffprobePath()` are aliases for `ensureFfmpeg()` /
`ensureFfprobe()`. Every call resolves the same cached path; only the first call
downloads.

## CLI use

The package installs two bins that print a resolved absolute path, so non-JS CI
steps can capture the binary without writing any JavaScript:

```sh
FFMPEG=$(npx nomercy-ffmpeg-path)
FFPROBE=$(npx nomercy-ffprobe-path)

"$FFMPEG" -i input.mkv -c copy output.mp4
```

The path is printed to stdout; errors go to stderr with a non-zero exit code.

## Environment overrides

| Variable | Effect |
| --- | --- |
| `NOMERCY_FFMPEG` | Absolute path to a pre-supplied `ffmpeg`. If it runs, it is returned as-is — no download. Use it to point at a CI cache or the MediaServer's bundled copy. |
| `NOMERCY_FFPROBE` | Same, for `ffprobe`. |
| `NOMERCY_FFMPEG_VERSION` | Override the pinned fork release tag (default `v1.0.36`). Lets CI bump the binary without a package release. |
| `NOMERCY_FFMPEG_FFVERSION` | Override the upstream ffmpeg version in the asset name (default `8.1.1`). |

Resolution order for each tool:

1. The `NOMERCY_FFMPEG` / `NOMERCY_FFPROBE` env var, if it points at a runnable binary.
2. A binary already cached from a previous run.
3. The matching platform artifact from the fork's GitHub release — downloaded,
   extracted, cached, and the archive cleaned up.

## Supported platforms

| OS | Arch | Release artifact |
| --- | --- | --- |
| Windows | x64 | `windows-x86_64.zip` |
| Windows | arm64 | `windows-aarch64.zip` |
| Linux | x64 | `linux-x86_64.tar.gz` |
| Linux | arm64 | `linux-aarch64.tar.gz` |
| macOS | arm64 | `darwin-arm64.tar.gz` |
| macOS | x64 | `darwin-x86_64.tar.gz` |

On an unsupported platform, resolution throws with the supported list and a
pointer to `NOMERCY_FFMPEG`. Set that env var to use a binary you supply.

## License

MIT — see [LICENSE](https://github.com/NoMercy-Entertainment/nomercy-ffmpeg/blob/master/LICENSE)
in the repository root.
