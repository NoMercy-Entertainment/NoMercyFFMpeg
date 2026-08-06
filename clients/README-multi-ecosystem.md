# NoMercy FFmpeg fork — multi-ecosystem static clients

One fork, one set of release artifacts, one resolve algorithm — bound into every
ecosystem that needs ffmpeg. The npm client (`clients/npm`) is built. This spec
is the contract its NuGet and Packagist siblings implement so all three behave
identically. Greenlight required before building the siblings.

## The shared contract

Every client, regardless of language, implements the same four-step resolve:

1. **Env override.** If `NOMERCY_FFMPEG` (ffmpeg) / `NOMERCY_FFPROBE` (ffprobe)
   names a runnable binary, return it as-is — no download. This is how a CI
   cache or the MediaServer's bundled copy is fed in.
2. **Cache hit.** If a binary is already cached for the pinned fork version,
   return it.
3. **Download.** Otherwise fetch the matching platform artifact from the fork's
   GitHub release, extract it, cache it, delete the archive.
4. **Verify.** Spawn `<binary> -version` and confirm the banner before returning;
   throw on an unsupported platform with a message that points at the env
   override.

### Pinned version (constants, env-overridable)

| Constant | Default | Env override |
| --- | --- | --- |
| Fork release tag | `v1.0.36` | `NOMERCY_FFMPEG_VERSION` |
| Upstream ffmpeg version | `8.1.1` | `NOMERCY_FFMPEG_FFVERSION` |
| Source repo | `NoMercy-Entertainment/nomercy-ffmpeg` | — |

### Platform → artifact map (identical across clients)

| OS | Arch | slug | ext | binaries |
| --- | --- | --- | --- | --- |
| Windows | x64 | `windows-x86_64` | `zip` | `ffmpeg.exe`, `ffprobe.exe` |
| Windows | arm64 | `windows-aarch64` | `zip` | `ffmpeg.exe`, `ffprobe.exe` |
| Linux | x64 | `linux-x86_64` | `tar.gz` | `ffmpeg`, `ffprobe` |
| Linux | arm64 | `linux-aarch64` | `tar.gz` | `ffmpeg`, `ffprobe` |
| macOS | arm64 | `darwin-arm64` | `tar.gz` | `ffmpeg`, `ffprobe` |
| macOS | x64 | `darwin-x86_64` | `tar.gz` | `ffmpeg`, `ffprobe` |

### Asset URL

```
https://github.com/NoMercy-Entertainment/nomercy-ffmpeg/releases/download/{TAG}/ffmpeg-{FFVERSION}-{slug}-{TAG}.{ext}
```

Example (Windows, pinned): `.../download/v1.0.36/ffmpeg-8.1.1-windows-x86_64-v1.0.36.zip`

Archives are flat: the binaries sit at the archive root. Extract straight into
the cache dir; no nested-directory stripping needed.

---

## NuGet sibling — `NoMercy.FFmpeg.Static`

For the .NET MediaServer startup. Resolves the fork binary once at boot and hands
the path to the encoding pipeline.

### Binding API

```csharp
namespace NoMercy.FFmpeg.Static;

public static class FFmpegStatic
{
    // Download-on-first-use, cached. Async over the network call.
    public static Task<string> EnsureFFmpegAsync(CancellationToken ct = default);
    public static Task<string> EnsureFFprobeAsync(CancellationToken ct = default);

    // Pure helpers — no I/O — for callers that want the URL/target.
    public static PlatformTarget ResolvePlatformTarget(OSPlatform os, Architecture arch);
    public static string AssetUrl(PlatformTarget target);
}
```

- **Env overrides:** read `NOMERCY_FFMPEG` / `NOMERCY_FFPROBE` /
  `NOMERCY_FFMPEG_VERSION` / `NOMERCY_FFMPEG_FFVERSION` via
  `Environment.GetEnvironmentVariable`.
- **Platform detect:** `RuntimeInformation.OSPlatform` + `.OSArchitecture`.
- **Download:** `HttpClient` with redirect follow (GitHub release → CDN).
- **Extract:** `System.IO.Compression.ZipFile` for `.zip`; shell out to `tar` (or
  `SharpZipLib`) for `.tar.gz`. Set the unix exec bit via `File.SetUnixFileMode`
  on non-Windows.
- **Cache root:** `Path.Combine(Path.GetTempPath(), "nomercy-ffmpeg-static", TAG)`,
  matching the npm client so a shared CI runner reuses one cache across
  ecosystems.
- **Verify:** `Process.Start(binary, "-version")`, assert exit 0 + banner.
- **Explicit types only** (`string`, `Task<string>`), never `var` — house C# rule.
  CSharpier before commit. `[GeneratedRegex]` for the banner check.

### Publish workflow shape

`.github/workflows/nuget-publish.yml`, mirroring the npm one:

- Triggers: `release: [created]` + `workflow_dispatch`.
- `dotnet build -c Release` → `dotnet test` → `dotnet pack`.
- Version-already-published guard: query the NuGet v3 flat index
  (`https://api.nuget.org/v3-flatcontainer/nomercy.ffmpeg.static/index.json`),
  skip if the package version is already listed.
- Push: `dotnet nuget push` with the **NuGet Trusted Publishing** OIDC flow
  (`NuGet/login` action → short-lived token), `id-token: write`, no long-lived
  `NUGET_API_KEY`. Mirrors the npm trusted-publishing posture.

### Compat note

The MediaServer already bundles its own ffmpeg. This package must **not** force a
download at startup when the bundled binary is present: the server sets
`NOMERCY_FFMPEG` to the bundled path, so step 1 short-circuits and nothing is
fetched. Wire it as an opt-in resolver, never a hard dependency that breaks an
offline self-hosted box.

---

## Packagist sibling — `nomercy/ffmpeg-static`

For Fillz's vidmount PHP website and any PHP CI step.

### Binding API

```php
namespace NoMercy\FFmpegStatic;

final class FFmpegStatic
{
    // Download-on-first-use, cached. Returns an absolute path.
    public static function ensureFfmpeg(): string;
    public static function ensureFfprobe(): string;

    // Pure helpers.
    public static function resolvePlatformTarget(?string $os = null, ?string $arch = null): PlatformTarget;
    public static function assetUrl(PlatformTarget $target): string;
}
```

- **Env overrides:** `getenv('NOMERCY_FFMPEG')` etc.
- **Platform detect:** `PHP_OS_FAMILY` (`Windows`/`Linux`/`Darwin`) +
  `php_uname('m')` (`x86_64`/`arm64`/`aarch64`).
- **Download:** Guzzle or a streamed `fopen`/`copy` with redirect follow.
- **Extract:** `ZipArchive` for `.zip`; `PharData` (or shell `tar`) for `.tar.gz`.
  `chmod(0o755)` the binary on non-Windows.
- **Cache root:** `sys_get_temp_dir() . '/nomercy-ffmpeg-static/' . TAG`.
- **Verify:** `proc_open` / `exec` `<binary> -version`, check the banner.
- **CLI bin:** a `bin/nomercy-ffmpeg-path` PHP script declared in
  `composer.json` `"bin"`, so `vendor/bin/nomercy-ffmpeg-path` prints the path.
- Target PHP 8.2+, `declare(strict_types=1)`, typed properties.

### Publish workflow shape

`.github/workflows/packagist-publish.yml`:

- Packagist publishes from a git tag via its GitHub webhook (auto-update on push
  of a new tag) — so the "workflow" is mainly a tag gate: on `release: [created]`,
  run `composer validate --strict`, `composer install`, the test suite, then push
  the tag. Packagist picks it up from the webhook; no API token in CI.
- Version guard: Packagist dedupes by tag, so re-tagging the same version is the
  no-op. CI asserts the `composer.json` version is not already a published tag
  before tagging.

---

## Why one package per ecosystem (not a wrapper)

Each ecosystem's consumers expect a native dependency they can `install` and a
native bin on `PATH` (`vendor/bin`, `node_modules/.bin`, a NuGet tool path).
Shelling out to a foreign runtime to resolve a binary defeats the point. The
resolve algorithm is ~120 lines; the cost of three small native ports is far
lower than the cost of a cross-runtime shim every consumer has to special-case.
The shared contract above is what keeps the three honest: same env vars, same
cache layout, same artifact map, same version pin.
