# FFmpeg multi-platform build (docker compose)

All platform services in `docker-compose.yml` pass `DEBUG=true` as a build arg,
so build logs stay verbose. `docker compose build` shows progress live with
`--progress=plain`; the per-stage CMD copies the final tarball to `./output/`
via the mounted volume.

> Tip: prepend `COMPOSE_PROGRESS=plain` (or pass `--progress=plain` per command)
> to force unbuffered, line-by-line output so failures are immediately visible.

```powershell
# 0. Make sure the host output directory exists (compose mounts ./output → /output)
New-Item -ItemType Directory -Force ./output | Out-Null
```

## 1. Base image (~30–60 min; downloads ~3 GB of deps)

```powershell
docker compose build --progress=plain ffmpeg-base
```

## 2. Fastest platform first for feedback (~45–90 min)

```powershell
# Build the image (DEBUG=true is wired in via docker-compose.yml build args)
docker compose build --progress=plain ffmpeg-linux-x86_64

# Run it once so the CMD copies ffmpeg-<ver>-linux-x86_64.tar.gz to ./output
docker compose run --rm ffmpeg-linux-x86_64
```

## 3. Verify every custom feature is present

```powershell
# Smoke-test the produced binary by extracting it from the build stage
docker compose run --rm --entrypoint ffmpeg ffmpeg-linux-x86_64 -version            # expect "9.0"
docker compose run --rm --entrypoint ffmpeg ffmpeg-linux-x86_64 -hide_banner -muxers   | Select-String "chapters_vtt|spritevtt|vobsub"
docker compose run --rm --entrypoint ffmpeg ffmpeg-linux-x86_64 -hide_banner -encoders | Select-String "ocr_subtitle"
docker compose run --rm --entrypoint ffmpeg ffmpeg-linux-x86_64 -hide_banner -filters  | Select-String "beatdetect"
```

> Note: the final image is `alpine` and its CMD just copies the tarball out;
> if `--entrypoint ffmpeg` does not resolve, run the verify step against the
> intermediate stage instead:
> `docker compose build --target linux ffmpeg-linux-x86_64`
> then `docker compose run --rm --entrypoint ffmpeg ffmpeg-linux-x86_64 -version`.

## 4. Smoke-test artifact name (proves parameterization works)

```powershell
# The compose CMD already copies the tarball to /output (mounted to ./output)
Get-ChildItem ./output/ffmpeg-*-linux-x86_64.tar.gz   # must exist
```

## 5. Functional tests

```powershell
# Linux/macOS / WSL
./tests/tests.sh "$(Get-Location)"

# Native Windows
./tests/tests.ps1
```

## 6. Remaining platforms (each ~45–120 min)

```powershell
docker compose build --progress=plain ffmpeg-linux-aarch64
docker compose run   --rm             ffmpeg-linux-aarch64

docker compose build --progress=plain ffmpeg-windows-x86_64
docker compose run   --rm             ffmpeg-windows-x86_64

# Windows on ARM. Builds locally but is not in the CI matrix and ships in no
# release — it has never been run on real hardware. See the "Windows on ARM"
# section in README.md before promoting it.
docker compose build --progress=plain ffmpeg-windows-aarch64
docker compose run   --rm             ffmpeg-windows-aarch64

docker compose build --progress=plain ffmpeg-darwin-x86_64
docker compose run   --rm             ffmpeg-darwin-x86_64

docker compose build --progress=plain ffmpeg-darwin-arm64
docker compose run   --rm             ffmpeg-darwin-arm64
```

## Build everything in one shot (uses depends_on order)

```powershell
docker compose build --progress=plain
docker compose up    --abort-on-container-exit
```

## Debugging a failing build

```powershell
# Re-run with no cache and plain progress so every command + stderr is visible
docker compose build --no-cache --progress=plain ffmpeg-linux-x86_64

# Drop into a shell at the failure point (uses the base image)
docker compose run --rm --entrypoint /bin/bash ffmpeg-base

# Tail the in-container build log (path used inside the dockerfiles)
docker compose run --rm --entrypoint /bin/bash ffmpeg-linux-x86_64 -c "cat /ffmpeg_build.log"
```
