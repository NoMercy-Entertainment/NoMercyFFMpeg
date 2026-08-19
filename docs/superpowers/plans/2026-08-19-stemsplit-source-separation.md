# stemsplit Source Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native FFmpeg audio filter `stemsplit` that separates music into `vocals` and `accompaniment` stems using the Deezer Spleeter 2stems model, running on the ggml runtime already linked into this repo's binaries.

**Architecture:** An offline Dockerised converter turns the TensorFlow checkpoint into a fp16 GGUF file plus numeric parity fixtures. The filter (`libavfilter/af_stemsplit.c`) buffers audio into 512-frame STFT segments using `av_tx`, runs one ggml U-Net per instrument, converts the two magnitude estimates into ratio masks, applies them to the original complex STFT, and resynthesises by weighted overlap-add. A build script patches the filter into FFmpeg the same way `scripts/57-keydetect.sh` does.

**Tech Stack:** C (FFmpeg 9.0 libavfilter), ggml (via whisper.cpp's static libs), `av_tx` FFT, Python 3 + TensorFlow 2 (converter only, Dockerised), Bash build scripts, Docker.

**Spec:** `docs/superpowers/specs/2026-08-19-stemsplit-source-separation-design.md` — read it before starting. Every constant in this plan comes from there; if the two ever disagree, the spec wins.

## Global Constraints

Every task's requirements implicitly include these.

- **FFmpeg version:** 9.0 (`ffmpeg_version=9.0` in `ffmpeg-base.dockerfile:15`).
- **Filter name:** `stemsplit`. Symbol `ff_af_stemsplit`. Config token `CONFIG_STEMSPLIT_FILTER`.
- **No new third-party dependency.** ggml comes from `whisper.pc`, already written by `scripts/48-whisper.sh`. If a task seems to need another library, stop and escalate.
- **Seven targets must keep building:** linux x86_64/aarch64, windows x86_64/aarch64, darwin x86_64/arm64, freebsd x86_64.
- **Model weights never enter git.** No file over 1 MB is committed. Fixtures are summary statistics, not full tensors.
- **Signal constants (must match exactly):** `sample_rate=44100`, `n_channels=2`, `frame_length=4096`, `frame_step=1024`, Hann **periodic** window, `T=512`, `F=1024`, full bins `2049`, `separation_exponent=2`, `EPSILON=1e-10`, `WINDOW_COMPENSATION_FACTOR=2.0/3.0`.
- **Commits:** Conventional Commits (`type(scope): description`). Commit messages end after the last content line — no trailing attribution or co-author footer of any kind.
- **Branch:** `feat/spleeter-filter`. Do not push unless explicitly asked.

### Tensor layout convention (used by every task from Task 6 onward)

ggml orders dimensions fastest-first: `ne = [W, H, C, N]`.

- Spectrogram tensors are `ne = [F, T, C, 1]` — so `ne[0]` is frequency, `ne[1]` is time.
- `ggml_conv_2d(ctx, kernel, input, s0, s1, p0, p1, d0, d1)` strides `s0` along `ne[0]` (frequency) and `s1` along `ne[1]` (time).
- Regular conv kernels are `ne = [KW, KH, IC, OC]`. TensorFlow stores `[kh, kw, ic, oc]`, so the converter swaps the first two axes.
- Transposed conv kernels are `ne = [KW, KH, OC, IC]`. TensorFlow stores `[kh, kw, oc, ic]`, so again only the first two axes swap.
- All conv kernels are stored **F16**; ggml's im2col path requires it. Bias, BatchNorm affine, and activations are F32.

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `tools/spleeter-gguf/Dockerfile` | create | TF2 image that reads the checkpoint. Never shipped. |
| `tools/spleeter-gguf/convert.py` | create | Checkpoint → fp16 GGUF. BN reduction, axis swaps, metadata. |
| `tools/spleeter-gguf/dump_reference.py` | create | Reference activations for one fixed segment → `fixtures.json`. |
| `tools/spleeter-gguf/compare.py` | create | Compares filter `dump=` output against `fixtures.json`. |
| `tools/spleeter-gguf/fixtures.json` | create | Committed. Per-layer shape + statistics + 64 sampled values. |
| `tools/spleeter-gguf/README.md` | create | How to regenerate the model and fixtures. |
| `scripts/includes/af_stemsplit.c` | create | The filter. Single file, matching `af_whisper.c` / `af_keydetect.c` convention. |
| `scripts/60-stemsplit.sh` | create | Patches the filter into the FFmpeg tree. |
| `tests/tests.sh` | modify | One `run_test` line. |
| `tests/tests.ps1` | modify | One `run_test` line. |
| `README.md` | modify | Custom-features table entry + usage. |

`af_stemsplit.c` stays one file. It will land around 1100–1300 lines, comparable to `af_keydetect.c` (1392 lines), and the repo's convention is one self-contained `.c` per custom filter — splitting it would break the `scripts/60-*.sh` copy pattern.

---

## Task 1: Build the iteration harness and settle the two ggml unknowns

The spec (section 6.3) defers two decisions to "step 1". This is that step. Nothing else can start until they are answered, because the model format depends on the first one.

**Files:**
- Create: `<scratchpad>/ffstem-vol.dockerfile` (throwaway, not committed)
- Create: `<scratchpad>/probe.c` (throwaway, not committed)

`<scratchpad>` is the session scratchpad directory. These files are deliberately **not** committed — prior filter work in this repo used the same throwaway-harness approach.

**Interfaces:**
- Consumes: nothing.
- Produces: a Docker image `ffstem-vol:latest` containing a configured FFmpeg 9.0 source tree at `/build/ffmpeg` with all dependencies (including whisper/ggml) installed at `${PREFIX}`, plus two written findings recorded in the task's commit message:
  - `HAS_GGUF_H` — whether `<gguf.h>` is installed and usable.
  - `HAS_CONV2D_DIRECT` — whether `ggml_conv_2d_direct` exists.

- [ ] **Step 1: Write the harness Dockerfile**

It reuses the repo's own build scripts, stopping *before* `rm -rf /build/ffmpeg` so the source tree survives for repeated `make` runs.

```dockerfile
# <scratchpad>/ffstem-vol.dockerfile
ARG BASE_TAG=latest
FROM nomercyentertainment/ffmpeg-base:${BASE_TAG} AS harness

ENV TARGET_OS=linux ARCH=x86_64 DEBUG=true

COPY ./scripts /scripts
RUN find /scripts -type f -name "*.sh" -exec sed -i 's/\r$//' {} +
RUN touch /build/enable.txt /build/cflags.txt /build/ldflags.txt /build/extra_libflags.txt

# Only the dependencies stemsplit actually needs: zlib, fftw, whisper (=> ggml).
RUN chmod +x /scripts/init/run-step.sh \
    && /scripts/init/run-step.sh 03-zlib.sh \
    && /scripts/init/run-step.sh 04-fftw3.sh \
    && /scripts/init/run-step.sh 48-whisper.sh

WORKDIR /build/ffmpeg
RUN FFMPEG_ENABLES=$(cat /build/enable.txt) export FFMPEG_ENABLES \
    && ./configure --prefix=${PREFIX} --disable-shared --enable-static \
       --enable-gpl --enable-version3 --enable-nonfree ${FFMPEG_ENABLES} \
       --enable-filter=all --disable-doc \
       --extra-libs="-lpthread -lm $(cat /build/extra_libflags.txt)" \
    && make -j$(nproc) ffmpeg
```

- [ ] **Step 2: Build the harness image**

Run from the repo root (the base image must exist; build it with `docker compose build ffmpeg-base` if it does not):

```bash
docker build -f "<scratchpad>/ffstem-vol.dockerfile" -t ffstem-vol:latest .
```

Expected: image builds, ending with a working `/build/ffmpeg/ffmpeg`. This takes 20–40 minutes once; every later iteration reuses it.

- [ ] **Step 3: Verify the harness produces a runnable binary**

```bash
docker run --rm ffstem-vol:latest /build/ffmpeg/ffmpeg -hide_banner -version
```

Expected: prints `ffmpeg version 9.0`.

- [ ] **Step 4: Write the ggml capability probe**

```c
/* <scratchpad>/probe.c */
#include <stdio.h>
#include <ggml.h>

#ifdef PROBE_GGUF
#include <gguf.h>
#endif

int main(void)
{
#ifdef PROBE_GGUF
    struct gguf_init_params p = { .no_alloc = true, .ctx = NULL };
    (void) gguf_init_from_file("/nonexistent.gguf", p);
    printf("gguf.h OK\n");
#endif
#ifdef PROBE_DIRECT
    (void) ggml_conv_2d_direct;
    printf("conv_2d_direct OK\n");
#endif
    printf("ggml OK\n");
    return 0;
}
```

- [ ] **Step 5: Run the probe twice and record both answers**

```bash
docker run --rm -v "<scratchpad>:/probe" ffstem-vol:latest sh -c '
  cd /probe
  gcc -DPROBE_GGUF probe.c -I${PREFIX}/include -L${PREFIX}/lib \
      -lggml -lggml-base -lggml-cpu -lstdc++ -lm -lpthread -o probe_gguf \
      && ./probe_gguf || echo "GGUF UNAVAILABLE"
  gcc -DPROBE_DIRECT probe.c -I${PREFIX}/include -L${PREFIX}/lib \
      -lggml -lggml-base -lggml-cpu -lstdc++ -lm -lpthread -o probe_direct \
      && ./probe_direct || echo "CONV2D_DIRECT UNAVAILABLE"
'
```

Expected: two definitive lines. Both outcomes are acceptable — this step decides, it does not fail.

**Act on the answers:**
- If `gguf.h` is **unavailable**, the model container becomes a flat binary instead of GGUF. Everywhere this plan says "GGUF", write: 12-byte header (`magic "STMS"`, `uint32 version=1`, `uint32 n_tensors`), then per tensor a `uint32 name_len`, name bytes, `uint32 n_dims`, `uint32 ne[4]`, `uint32 ggml_type`, then the raw data 32-byte aligned. Metadata keys become a leading JSON blob with its own `uint32` length. Nothing else in the plan changes.
- If `ggml_conv_2d_direct` is **unavailable**, use `ggml_conv_2d` with F16 kernels — which is what Task 7 assumes anyway.

- [ ] **Step 6: Record the findings in the spec**

Append a short subsection to `docs/superpowers/specs/2026-08-19-stemsplit-source-separation-design.md` under section 6.3 stating both answers and the date they were probed, so no later reader has to re-derive them.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/specs/2026-08-19-stemsplit-source-separation-design.md
git commit -m "docs(stemsplit): record ggml capability probe results

Probed the pinned whisper.cpp 1.9.1 ggml for gguf.h availability and
ggml_conv_2d_direct, the two runtime unknowns the design deferred.
Records both answers so the model container format is settled before
any implementation depends on it."
```

---

## Task 2: Model converter — weights to GGUF

**Files:**
- Create: `tools/spleeter-gguf/Dockerfile`
- Create: `tools/spleeter-gguf/convert.py`
- Create: `tools/spleeter-gguf/README.md`

**Interfaces:**
- Consumes: the container format decided in Task 1.
- Produces: `spleeter-2stems-f16.gguf` with these exact tensor names, two networks named `vocals` and `accompaniment`:
  - `<inst>.conv<N>.weight` — F16, `[5, 5, IC, OC]`, `N` in 1..6
  - `<inst>.conv<N>.bias` — F32, `[OC]`
  - `<inst>.conv<N>.bn_a`, `<inst>.conv<N>.bn_b` — F32, `[OC]`
  - `<inst>.up<N>.weight` — F16, `[5, 5, OC, IC]`, `N` in 1..6
  - `<inst>.up<N>.bias` — F32, `[OC]`
  - `<inst>.up<N>.bn_a`, `<inst>.up<N>.bn_b` — F32, `[OC]`
  - `<inst>.out.weight` — F16, `[4, 4, 1, 2]`
  - `<inst>.out.bias` — F32, `[2]`
  - Metadata: `stemsplit.arch="spleeter-unet-v1"`, `stemsplit.sample_rate=44100`, `stemsplit.frame_length=4096`, `stemsplit.frame_step=1024`, `stemsplit.n_t=512`, `stemsplit.n_f=1024`, `stemsplit.separation_exponent=2`, `stemsplit.instruments=["vocals","accompaniment"]`
- Note: `up6` has `OC=1`; there is **no** `bn_a`/`bn_b` for `out` (no BatchNorm follows the sigmoid conv).

- [ ] **Step 1: Write the converter Dockerfile**

```dockerfile
# tools/spleeter-gguf/Dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir tensorflow-cpu==2.16.2 numpy==1.26.4 gguf==0.10.0

WORKDIR /work
COPY convert.py dump_reference.py compare.py /work/

# The 2stems checkpoint, MIT-licensed, from the spleeter release assets.
ADD https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz /work/2stems.tar.gz
RUN mkdir -p /work/2stems && tar -xzf 2stems.tar.gz -C /work/2stems && rm 2stems.tar.gz

ENTRYPOINT ["python"]
```

- [ ] **Step 2: Write the failing test — a shape assertion script**

The converter's test is that the emitted file has exactly the expected tensors with the expected shapes. Put the assertion in `convert.py` itself as a `--verify` mode so it cannot drift from the writer.

```python
# tools/spleeter-gguf/convert.py  (test portion, written first)
EXPECTED = {}
for inst in ("vocals", "accompaniment"):
    for n, (ic, oc) in enumerate([(2,16),(16,32),(32,64),(64,128),(128,256),(256,512)], 1):
        EXPECTED[f"{inst}.conv{n}.weight"] = (5, 5, ic, oc)
        EXPECTED[f"{inst}.conv{n}.bias"]   = (oc,)
        EXPECTED[f"{inst}.conv{n}.bn_a"]   = (oc,)
        EXPECTED[f"{inst}.conv{n}.bn_b"]   = (oc,)
    for n, (ic, oc) in enumerate([(512,256),(512,128),(256,64),(128,32),(64,16),(32,1)], 1):
        EXPECTED[f"{inst}.up{n}.weight"] = (5, 5, oc, ic)
        EXPECTED[f"{inst}.up{n}.bias"]   = (oc,)
        EXPECTED[f"{inst}.up{n}.bn_a"]   = (oc,)
        EXPECTED[f"{inst}.up{n}.bn_b"]   = (oc,)
    EXPECTED[f"{inst}.out.weight"] = (4, 4, 1, 2)
    EXPECTED[f"{inst}.out.bias"]   = (2,)


def verify(tensors):
    missing = sorted(set(EXPECTED) - set(tensors))
    extra   = sorted(set(tensors) - set(EXPECTED))
    assert not missing, f"missing tensors: {missing}"
    assert not extra, f"unexpected tensors: {extra}"
    for name, shape in EXPECTED.items():
        got = tuple(tensors[name].shape)
        assert got == shape, f"{name}: expected {shape}, got {got}"
    print(f"verify OK: {len(EXPECTED)} tensors")
```

Note the decoder channel counts: `up1` takes 512 (conv6) and emits 256; `up2` takes 512 (up1's 256 concatenated with conv5's 256) and emits 128; and so on down to `up6`, which takes 32 and emits 1.

- [ ] **Step 3: Run the verification against an empty dict to confirm it fails**

```bash
docker build -t spleeter-gguf tools/spleeter-gguf
docker run --rm spleeter-gguf -c "import convert; convert.verify({})"
```

Expected: `AssertionError: missing tensors: [...]` listing 100 names.

- [ ] **Step 4: Implement the conversion**

```python
# tools/spleeter-gguf/convert.py  (implementation portion)
import numpy as np
import tensorflow as tf

BN_EPS = 1e-3  # Keras BatchNormalization default

def bn_affine(reader, prefix):
    """Reduce a BatchNorm to per-channel y = a*x + b."""
    gamma = reader.get_tensor(f"{prefix}/gamma")
    beta  = reader.get_tensor(f"{prefix}/beta")
    mean  = reader.get_tensor(f"{prefix}/moving_mean")
    var   = reader.get_tensor(f"{prefix}/moving_variance")
    a = gamma / np.sqrt(var + BN_EPS)
    b = beta - mean * a
    return a.astype(np.float32), b.astype(np.float32)


def conv_weight(reader, name, transposed):
    """TF [kh,kw,ic,oc] (or [kh,kw,oc,ic]) -> ggml [kw,kh,...] as F16."""
    w = reader.get_tensor(name)
    w = np.transpose(w, (1, 0, 2, 3))  # swap kh/kw only
    return np.ascontiguousarray(w).astype(np.float16)
```

Walk the checkpoint's variable list with `tf.train.load_checkpoint(ckpt_dir)` and `reader.get_variable_to_shape_map()`. Spleeter's graph names the two networks in the order given by `instrument_list`, so map them to `vocals` and `accompaniment` by that order and **print the mapping** so a human can sanity-check it. Call `verify()` before writing, then write via the `gguf` package's `GGUFWriter` (or the flat container from Task 1 if GGUF is unavailable).

- [ ] **Step 5: Run the conversion and confirm verification passes**

```bash
docker run --rm -v "$PWD/out:/out" spleeter-gguf convert.py \
    --checkpoint /work/2stems --output /out/spleeter-2stems-f16.gguf
```

Expected: `verify OK: 100 tensors`, then a written file. Confirm the size:

```bash
ls -l out/spleeter-2stems-f16.gguf
```

Expected: roughly 39 MB (two networks of ~9.82 M fp16 parameters).

- [ ] **Step 6: Write the converter README**

`tools/spleeter-gguf/README.md` documents: the build and run commands above, that the output is a release asset and must never be committed, the MIT licence of the Spleeter code and checkpoints, the JOSS citation (Hennequin et al., 2020, doi:10.21105/joss.02154), and upstream's advisory that users must hold rights to any copyrighted material they process.

- [ ] **Step 7: Commit**

```bash
git add tools/spleeter-gguf/
git commit -m "feat(stemsplit): add spleeter checkpoint to GGUF converter

Dockerised TensorFlow image that reads the MIT-licensed 2stems
checkpoint and emits fp16 weights in the layout the filter expects:
kh/kw axes swapped for ggml, and every BatchNorm reduced to a
per-channel affine so the decoder's post-activation norms need no
special case at runtime.

Verifies all 100 tensor names and shapes before writing. Output is a
release asset and is never committed."
```

---

## Task 3: Model converter — reference fixtures

Without these, Task 7 and Task 8 have nothing to test against, and the padding arithmetic (the spec's highest risk) goes unverified.

**Files:**
- Create: `tools/spleeter-gguf/dump_reference.py`
- Create: `tools/spleeter-gguf/compare.py`
- Create: `tools/spleeter-gguf/fixtures.json`

**Interfaces:**
- Consumes: the checkpoint and the tensor naming from Task 2.
- Produces:
  - `fixtures.json` — committed, under 200 KB. Keys are layer ids (`vocals.conv1`, `vocals.up3`, `vocals.out`, and the same for `accompaniment`; 26 entries). Each value is `{"shape": [F, T, C], "mean": f, "std": f, "min": f, "max": f, "samples": [[i, j, c, value], ...]}` with exactly 64 pseudo-random but **fixed** sample positions (seeded `numpy.random.default_rng(20260819)`).
  - `compare.py` — CLI: `compare.py --fixtures fixtures.json --dump <dir> [--rtol 1e-3]`. Reads `<dir>/<layer>.f32` raw float32 files written by the filter's `dump=` option, in `[F, T, C]` C-order, and exits non-zero listing every layer that fails.
- The reference input segment is deterministic and needs no audio file: a fixed pseudo-random magnitude spectrogram, `numpy.random.default_rng(20260819).random((512, 1024, 2), dtype=np.float32) * 4.0`, written to `fixtures_input.f32` alongside the dump directory.

- [ ] **Step 1: Write `dump_reference.py`**

Rebuild the U-Net in Keras exactly as `spleeter/model/functions/unet.py` defines it — encoder `Conv2D(5,5, strides=2, padding="same")` → `BatchNormalization` → `LeakyReLU(0.2)`; decoder `Conv2DTranspose(5,5, strides=2, padding="same")` → `ReLU` → `BatchNormalization` → concat; output `Conv2D(2, (4,4), dilation_rate=(2,2), padding="same", activation="sigmoid")` then multiply by the input. Load the checkpoint weights into it, run the fixed input, and capture every intermediate.

Note the ordering carefully: **encoder normalises after the convolution and before the activation; decoder normalises after the activation.** Getting this backwards produces a reference that is itself wrong, which would then "validate" a wrong implementation.

- [ ] **Step 2: Run it and confirm 26 layers are captured**

```bash
docker run --rm -v "$PWD/out:/out" spleeter-gguf dump_reference.py \
    --checkpoint /work/2stems --fixtures /out/fixtures.json --raw-dir /out/ref
```

Expected: `wrote 26 layer fixtures` and a `fixtures.json` under 200 KB.

- [ ] **Step 3: Write `compare.py` and prove it detects a one-sample shift**

This is the test *of the test* — a comparator that cannot catch the shift bug is worthless.

```bash
docker run --rm -v "$PWD/out:/out" spleeter-gguf -c "
import numpy as np, json, subprocess, sys
ref = np.fromfile('/out/ref/vocals.conv1.f32', dtype=np.float32).reshape(512, 256, 16)
shifted = np.roll(ref, 1, axis=0)
shifted.tofile('/out/bad/vocals.conv1.f32')
"
docker run --rm -v "$PWD/out:/out" spleeter-gguf compare.py \
    --fixtures /out/fixtures.json --dump /out/bad
```

Expected: exit code 1 and a line naming `vocals.conv1` as failing. If it passes, the sample positions are too forgiving — increase the sample count or include the min/max checks in the verdict.

- [ ] **Step 4: Confirm the comparator passes on the unmodified reference**

```bash
docker run --rm -v "$PWD/out:/out" spleeter-gguf compare.py \
    --fixtures /out/fixtures.json --dump /out/ref
```

Expected: exit code 0, `26/26 layers match`.

- [ ] **Step 5: Commit**

```bash
git add tools/spleeter-gguf/dump_reference.py tools/spleeter-gguf/compare.py tools/spleeter-gguf/fixtures.json
git commit -m "test(stemsplit): add per-layer parity fixtures and comparator

Captures all 26 layer outputs of both networks for a fixed synthetic
input segment, stored as shape, statistics and 64 fixed sample points
so the committed fixture stays under 200 KB.

The comparator is itself tested against a one-sample-shifted tensor,
which is the exact failure mode the asymmetric SAME padding can
produce and the reason these fixtures exist."
```

---

## Task 4: Filter skeleton, build integration, passthrough

Deliberately does no signal processing. Its deliverable is that the filter exists, registers, negotiates formats, and passes audio through untouched — so every later task starts from a known-good build.

**Files:**
- Create: `scripts/includes/af_stemsplit.c`
- Create: `scripts/60-stemsplit.sh`
- Modify: `tests/tests.sh:283` area (add one `run_test` line)
- Modify: `tests/tests.ps1:220` area (add one `run_test` line)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `typedef struct StemSplitContext` with fields `const AVClass *class; char *model_path; int stem; int highband; int64_t overlap; int nb_threads; char *dump_dir; int eof; int64_t next_pts;`
  - `const FFFilter ff_af_stemsplit`
  - Option constants: `SS_STEM_VOCALS=0`, `SS_STEM_ACCOMPANIMENT=1`, `SS_STEM_ALL=2`; `SS_HB_PASSTHROUGH=0`, `SS_HB_ZEROS=1`, `SS_HB_AVERAGE=2`

- [ ] **Step 1: Write the failing test**

Add to `tests/tests.sh`, immediately after the `libdav1d` line:

```bash
run_test "stemsplit" "-hide_banner -filters | grep stemsplit" "stemsplit"
```

And the matching line in `tests/tests.ps1`, after its `libdav1d` line:

```powershell
run_test "stemsplit" "-hide_banner -filters | findstr stemsplit" "stemsplit"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
docker run --rm ffstem-vol:latest /build/ffmpeg/ffmpeg -hide_banner -filters | grep stemsplit
```

Expected: no output, exit code 1 — the filter does not exist yet.

- [ ] **Step 3: Write the filter skeleton**

`scripts/includes/af_stemsplit.c`, opening with the standard FFmpeg LGPL 2.1-or-later header block copied from `scripts/includes/af_whisper.c:1-19`.

```c
#include <ggml.h>

#include "libavutil/opt.h"
#include "libavutil/channel_layout.h"
#include "libavutil/samplefmt.h"
#include "libavutil/mem.h"
#include "libavutil/tx.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/audio.h"
#include "libavfilter/filters.h"
#include "formats.h"

#define SS_SAMPLE_RATE 44100
#define SS_FRAME_LENGTH 4096
#define SS_FRAME_STEP   1024
#define SS_BINS         (SS_FRAME_LENGTH / 2 + 1)   /* 2049 */
#define SS_T            512
#define SS_F            1024
#define SS_CHANNELS     2
#define SS_EPSILON      1e-10f
#define SS_WINDOW_COMPENSATION (2.0f / 3.0f)

enum { SS_STEM_VOCALS, SS_STEM_ACCOMPANIMENT, SS_STEM_ALL };
enum { SS_HB_PASSTHROUGH, SS_HB_ZEROS, SS_HB_AVERAGE };

typedef struct StemSplitContext {
    const AVClass *class;
    char    *model_path;
    int      stem;
    int      highband;
    int64_t  overlap;
    int      nb_threads;
    char    *dump_dir;

    int      eof;
    int64_t  next_pts;
} StemSplitContext;

static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_NONE };
    AVChannelLayout chlayouts[] = { AV_CHANNEL_LAYOUT_STEREO, { 0 } };
    int sample_rates[] = { SS_SAMPLE_RATE, -1 };
    int ret;

    if ((ret = ff_set_common_formats_from_list2(ctx, cfg_in, cfg_out, sample_fmts)) < 0)
        return ret;
    if ((ret = ff_set_common_channel_layouts_from_list2(ctx, cfg_in, cfg_out, chlayouts)) < 0)
        return ret;
    return ff_set_common_samplerates_from_list2(ctx, cfg_in, cfg_out, sample_rates);
}
```

Add `init()` (validate `model_path` is set, return `AVERROR(EINVAL)` naming the option if not), `uninit()`, a `filter_frame()` that forwards the frame unchanged, an `activate()` copied in shape from `af_whisper.c:463-500`, the `AVOption` table matching the spec's section 8, the `AVClass`, and:

```c
const FFFilter ff_af_stemsplit = {
    .p.name        = "stemsplit",
    .p.description = NULL_IF_CONFIG_SMALL("Separate music into vocal and accompaniment stems."),
    .p.priv_class  = &stemsplit_class,
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    .priv_size     = sizeof(StemSplitContext),
    FILTER_INPUTS(ff_audio_default_filterpad),
    FILTER_OUTPUTS(ff_audio_default_filterpad),
    FILTER_QUERY_FUNC2(query_formats),
};
```

The `stem` option is accepted but ignored in this task; a single output pad is used. Task 9 makes it real.

- [ ] **Step 4: Write the build script**

`scripts/60-stemsplit.sh`, structurally identical to `scripts/57-keydetect.sh` — copy that file and change the names. Four verified steps:

```bash
cp /scripts/includes/af_stemsplit.c /build/ffmpeg/libavfilter/af_stemsplit.c

# 1. allfilters.c
sed -i '0,/^extern const FFFilter ff_af_volumedetect;$/s//&\nextern const FFFilter ff_af_stemsplit;/' \
    /build/ffmpeg/libavfilter/allfilters.c

# 2. Makefile
sed -i '/^OBJS-\$(CONFIG_ABENCH_FILTER)/a\
OBJS-$(CONFIG_STEMSPLIT_FILTER)          += af_stemsplit.o' \
    /build/ffmpeg/libavfilter/Makefile

# 3. configure
sed -i '/^abench_filter_deps=/i stemsplit_filter_deps="whisper"' /build/ffmpeg/configure
```

Each guarded by a `grep -q` idempotence check and followed by a verification `grep` that `exit 1`s on failure, exactly as `57-keydetect.sh` does.

- [ ] **Step 5: Build and run the test**

```bash
docker run --rm -v "$PWD/scripts:/scripts" ffstem-vol:latest sh -c '
  /scripts/60-stemsplit.sh && cd /build/ffmpeg && ./configure --enable-filter=all >/dev/null \
    && make -j$(nproc) ffmpeg && ./ffmpeg -hide_banner -filters | grep stemsplit'
```

Expected: a line containing `stemsplit`.

- [ ] **Step 6: Verify passthrough is bit-exact**

```bash
docker run --rm -v "$PWD/scripts:/scripts" ffstem-vol:latest sh -c '
  cd /build/ffmpeg
  ./ffmpeg -y -f lavfi -i "sine=frequency=440:duration=3:sample_rate=44100" \
     -ac 2 -c:a pcm_s16le /tmp/in.wav
  ./ffmpeg -y -i /tmp/in.wav -af "stemsplit=model=/dev/null" -c:a pcm_s16le /tmp/out.wav
  cmp /tmp/in.wav /tmp/out.wav && echo PASSTHROUGH-OK'
```

Expected: `PASSTHROUGH-OK`. (`model=/dev/null` is accepted at this stage because nothing loads it yet; Task 6 makes it an error.)

- [ ] **Step 7: Commit**

```bash
git add scripts/includes/af_stemsplit.c scripts/60-stemsplit.sh tests/tests.sh tests/tests.ps1
git commit -m "feat(stemsplit): add filter skeleton and build integration

Registers the stemsplit filter, negotiates 44.1 kHz stereo planar
float, and passes audio through unchanged. The build script follows
the keydetect pattern: copy the source, patch allfilters.c and the
Makefile, and declare the whisper dependency that carries libggml.

Smoke coverage added to both the bash and PowerShell suites."
```

---

## Task 5: STFT and inverse STFT

**Files:**
- Modify: `scripts/includes/af_stemsplit.c`

**Interfaces:**
- Consumes: `StemSplitContext` from Task 4.
- Produces, added to `StemSplitContext`: `AVTXContext *tx_fwd, *tx_inv; av_tx_fn tx_fwd_fn, tx_inv_fn; float *window; float *in_buf; float *ola_buf; AVComplexFloat *spec;` and these functions:
  - `static int ss_dsp_init(AVFilterContext *ctx)` — returns 0 or `AVERROR(ENOMEM)`
  - `static void ss_dsp_free(StemSplitContext *s)`
  - `static void ss_analyze(StemSplitContext *s, const float *in, AVComplexFloat *out)` — `in` is `SS_FRAME_LENGTH` samples of one channel, `out` receives `SS_BINS` bins
  - `static void ss_synthesize(StemSplitContext *s, const AVComplexFloat *in, float *ola, int offset)` — windows, scales by `SS_WINDOW_COMPENSATION`, and overlap-adds into `ola` at `offset`

- [ ] **Step 1: Write the failing test**

A round-trip check driven through the filter. Add a temporary internal option `passthrough_dsp` (removed in Task 9) that runs analysis then synthesis with no mask, so the test can be an ordinary ffmpeg invocation measuring the residual:

```bash
docker run --rm ffstem-vol:latest sh -c '
  cd /build/ffmpeg
  ./ffmpeg -y -f lavfi -i "sine=frequency=1000:duration=5:sample_rate=44100" -ac 2 /tmp/a.wav
  ./ffmpeg -y -i /tmp/a.wav -af "stemsplit=model=/dev/null:passthrough_dsp=1" /tmp/b.wav
  ./ffmpeg -i /tmp/a.wav -i /tmp/b.wav -filter_complex \
     "[1:a]aeval=-val(0)|-val(1)[neg];[0:a][neg]amix=inputs=2:weights=1 1,volumedetect" \
     -f null - 2>&1 | grep max_volume'
```

Expected once implemented: `max_volume` at or below `-90.0 dB`.

- [ ] **Step 2: Run it and confirm it fails**

Expected now: the option does not exist — `Option 'passthrough_dsp' not found`.

- [ ] **Step 3: Implement the DSP**

```c
static int ss_dsp_init(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    float scale = 1.0f;
    int ret;

    s->window = av_malloc_array(SS_FRAME_LENGTH, sizeof(*s->window));
    if (!s->window)
        return AVERROR(ENOMEM);
    /* Periodic Hann, matching tf.signal.hann_window(periodic=True). */
    for (int n = 0; n < SS_FRAME_LENGTH; n++)
        s->window[n] = 0.5f - 0.5f * cosf(2.0f * M_PI * n / SS_FRAME_LENGTH);

    ret = av_tx_init(&s->tx_fwd, &s->tx_fwd_fn, AV_TX_FLOAT_RDFT, 0,
                     SS_FRAME_LENGTH, &scale, 0);
    if (ret < 0)
        return ret;

    scale = 1.0f / SS_FRAME_LENGTH;
    return av_tx_init(&s->tx_inv, &s->tx_inv_fn, AV_TX_FLOAT_RDFT, 1,
                      SS_FRAME_LENGTH, &scale, 0);
}
```

`ss_analyze` multiplies the input frame by `s->window` into a scratch buffer and calls `s->tx_fwd_fn`. `ss_synthesize` calls `s->tx_inv_fn`, multiplies by `s->window` again, scales by `SS_WINDOW_COMPENSATION`, and adds into `ola`.

The lead-in and crop are part of the segmenter: the filter feeds `SS_FRAME_LENGTH` zeros before the first real sample and discards the first `SS_FRAME_LENGTH` output samples, so the round trip is aligned. Verify the sign convention of `AV_TX_FLOAT_RDFT` output packing against `libavutil/tx.h` before assuming bin ordering; the round-trip test catches it if wrong.

- [ ] **Step 4: Run the round-trip test**

Expected: `max_volume: -90.0 dB` or lower. If it lands near `-6 dB`, the compensation factor is being applied twice or not at all; if near `0 dB` with a delay, the lead-in crop is wrong.

- [ ] **Step 5: Commit**

```bash
git add scripts/includes/af_stemsplit.c
git commit -m "feat(stemsplit): add STFT analysis and weighted overlap-add synthesis

4096-point av_tx RDFT with a periodic Hann window at hop 1024, matching
tf.signal.stft. Synthesis applies the window a second time and scales by
2/3, the exact reconstruction gain for Hann at 75 percent overlap.

Verified by round-trip: analysis followed by synthesis with no mask
reconstructs to below -90 dB."
```

---

## Task 6: Model loading

**Files:**
- Modify: `scripts/includes/af_stemsplit.c`

**Interfaces:**
- Consumes: the tensor names and metadata keys from Task 2.
- Produces:
  - `typedef struct SSLayer { struct ggml_tensor *w, *b, *bn_a, *bn_b; } SSLayer;`
  - `typedef struct SSNet { SSLayer conv[6], up[6], out; } SSNet;`
  - Added to `StemSplitContext`: `struct ggml_context *gguf_ctx; struct ggml_backend *backend; struct ggml_backend_buffer *weights_buf; SSNet nets[2]; int nb_instruments; char *instrument_names[2];`
  - `static int ss_model_load(AVFilterContext *ctx)` — 0, `AVERROR(EIO)`, or `AVERROR(EINVAL)`
  - `static void ss_model_free(StemSplitContext *s)`

- [ ] **Step 1: Write the failing tests — the six error paths**

```bash
docker run --rm ffstem-vol:latest sh -c '
  cd /build/ffmpeg
  fail() { echo "MISSING ERROR: $1"; }
  ./ffmpeg -f lavfi -i "sine=duration=1:sample_rate=44100" -ac 2 -af "stemsplit" -f null - 2>&1 \
     | grep -q "model" || fail "no-model-option"
  ./ffmpeg -f lavfi -i "sine=duration=1:sample_rate=44100" -ac 2 -af "stemsplit=model=/nope.gguf" -f null - 2>&1 \
     | grep -q "/nope.gguf" || fail "missing-file-path-in-message"
  ./ffmpeg -f lavfi -i "sine=duration=1:sample_rate=44100" -ac 2 -af "stemsplit=model=/etc/hostname" -f null - 2>&1 \
     | grep -qi "arch\|not a valid" || fail "bad-container"
  echo DONE'
```

Expected before implementation: three `MISSING ERROR:` lines.

- [ ] **Step 2: Implement loading**

Bridge ggml's logging to `av_log` first, using the `cb_log` pattern from `af_whisper.c:76-90`. Then:

```c
static int ss_model_load(AVFilterContext *ctx)
{
    StemSplitContext *s = ctx->priv;
    struct gguf_init_params gp = { .no_alloc = false, .ctx = &s->gguf_ctx };
    struct gguf_context *gguf;
    const char *arch;

    if (!s->model_path || !*s->model_path) {
        av_log(ctx, AV_LOG_ERROR,
               "No model specified. Use the 'model' option to point at a "
               "stemsplit .gguf file.\n");
        return AVERROR(EINVAL);
    }

    gguf = gguf_init_from_file(s->model_path, gp);
    if (!gguf) {
        av_log(ctx, AV_LOG_ERROR,
               "Could not read model '%s'. Download it from the releases page "
               "of this repository.\n", s->model_path);
        return AVERROR(EIO);
    }
    /* ... read stemsplit.arch, reject anything but "spleeter-unet-v1" ... */
}
```

Then resolve all 100 tensors by name, checking each `ne[]` against the expected shape and failing with `AVERROR(EINVAL)` naming the offending tensor. Read the instrument list from metadata into `instrument_names`. Call it from `init()` after option validation.

- [ ] **Step 3: Run the error-path tests**

Expected: `DONE` with no `MISSING ERROR:` lines.

- [ ] **Step 4: Verify the real model loads**

```bash
docker run --rm -v "$PWD/out:/models" ffstem-vol:latest sh -c '
  cd /build/ffmpeg
  ./ffmpeg -v verbose -f lavfi -i "sine=duration=1:sample_rate=44100" -ac 2 \
     -af "stemsplit=model=/models/spleeter-2stems-f16.gguf" -f null - 2>&1 \
     | grep -i "stemsplit.*loaded\|instruments"'
```

Expected: a log line naming both instruments, `vocals` and `accompaniment`.

- [ ] **Step 5: Commit**

```bash
git add scripts/includes/af_stemsplit.c
git commit -m "feat(stemsplit): load GGUF weights and validate the model

Resolves all 100 tensors by name and checks every shape against the
architecture, so a mismatched or truncated model fails at init with a
message naming the offending tensor rather than producing silence.

ggml log output is bridged to av_log at matching severity."
```

---

## Task 7: Encoder graph and padding

The highest-risk task. It exists separately from the decoder precisely so a reviewer can reject the padding arithmetic on its own.

**Files:**
- Modify: `scripts/includes/af_stemsplit.c`

**Interfaces:**
- Consumes: `SSNet`/`SSLayer` from Task 6, fixtures from Task 3.
- Produces:
  - `static struct ggml_tensor *ss_pad_tf(struct ggml_context *g, struct ggml_tensor *x, int before, int after)` — pads `ne[0]` and `ne[1]` with zeros, `before` at the low index and `after` at the high one
  - `static struct ggml_tensor *ss_bn(struct ggml_context *g, struct ggml_tensor *x, const SSLayer *l)`
  - `static struct ggml_tensor *ss_encoder_block(struct ggml_context *g, struct ggml_tensor *x, const SSLayer *l)` — pad(1,2) → `ggml_conv_2d(stride 2, pad 0)` → bias → BN → `ggml_leaky_relu(0.2)`
  - `static void ss_dump_tensor(StemSplitContext *s, const char *name, struct ggml_tensor *t)` — writes `<dump_dir>/<name>.f32` when `dump_dir` is set

- [ ] **Step 1: Write the failing test**

```bash
docker run --rm -v "$PWD/out:/models" ffstem-vol:latest sh -c '
  mkdir -p /tmp/dump && cd /build/ffmpeg
  ./ffmpeg -y -f lavfi -i "anoisesrc=duration=12:sample_rate=44100:seed=42" -ac 2 \
     -af "stemsplit=model=/models/spleeter-2stems-f16.gguf:dump=/tmp/dump" -f null -
  ls /tmp/dump'
docker run --rm -v "$PWD/out:/out" -v "/tmp/dump:/dump" spleeter-gguf compare.py \
    --fixtures /out/fixtures.json --dump /dump --layers "conv"
```

Expected once implemented: the 12 encoder layers (6 per network) match. `--layers conv` restricts the comparison to encoder layers so this task is not blocked on the decoder.

- [ ] **Step 2: Run it and confirm it fails**

Expected: `/tmp/dump` is empty — no dump support and no graph yet.

- [ ] **Step 3: Implement the padding helper**

```c
/* TensorFlow 'SAME' with kernel 5, stride 2 pads 1 before and 2 after on both
 * spatial axes. ggml_conv_2d only offers symmetric padding, so the asymmetry is
 * materialised here and the convolution then runs with pad 0.
 *
 * Symmetric pad 2 yields the correct output *shape* but shifts every window by
 * one input sample, which sounds plausible and is wrong. Do not "simplify" this.
 */
static struct ggml_tensor *ss_pad_tf(struct ggml_context *g,
                                     struct ggml_tensor *x,
                                     int before, int after)
{
    struct ggml_tensor *out = ggml_new_tensor_4d(g, x->type,
                                                 x->ne[0] + before + after,
                                                 x->ne[1] + before + after,
                                                 x->ne[2], x->ne[3]);
    out = ggml_scale(g, out, 0.0f);
    return ggml_acc(g, out, x,
                    out->nb[1], out->nb[2], out->nb[3],
                    before * out->nb[0] + before * out->nb[1]);
}
```

- [ ] **Step 4: Implement the encoder block and the dump hook**

```c
static struct ggml_tensor *ss_encoder_block(struct ggml_context *g,
                                            struct ggml_tensor *x,
                                            const SSLayer *l)
{
    x = ss_pad_tf(g, x, 1, 2);
    x = ggml_conv_2d(g, l->w, x, /*s0*/ 2, /*s1*/ 2, /*p0*/ 0, /*p1*/ 0, /*d0*/ 1, /*d1*/ 1);
    x = ggml_add(g, x, ggml_reshape_4d(g, l->b, 1, 1, l->b->ne[0], 1));
    x = ss_bn(g, x, l);
    return ggml_leaky_relu(g, x, 0.2f, false);
}
```

`ss_bn` is `ggml_add(g, ggml_mul(g, x, a4d), b4d)` with both reshaped to `[1, 1, C, 1]` so ggml broadcasts them across frequency and time.

- [ ] **Step 5: Assert the shapes before trusting the values**

Add a `av_assert0` after each encoder block that `x->ne[0]` and `x->ne[1]` halved and `x->ne[2] == expected_channels`. Run the dump command; expected shapes are `[512,256,16]`, `[256,128,32]`, `[128,64,64]`, `[64,32,128]`, `[32,16,256]`, `[16,8,512]` in `[F,T,C]` order.

- [ ] **Step 6: Run the parity comparison**

Expected: `12/12 layers match`. A failure on `conv1` alone means the input spectrogram is wrong; a failure that starts at `conv1` and grows means the padding offset is wrong; matching statistics with failing sample points means an axis transposition.

- [ ] **Step 7: Commit**

```bash
git add scripts/includes/af_stemsplit.c
git commit -m "feat(stemsplit): add the encoder graph with TensorFlow SAME padding

Six strided convolution blocks built on ggml, each preceded by an
explicit asymmetric pad of 1 before and 2 after, because TensorFlow's
SAME padding is asymmetric for kernel 5 stride 2 and ggml only offers
symmetric padding. The symmetric shortcut yields the right output shape
with every window shifted one sample, so it is called out in a comment.

All six encoder layers of both networks match the Python reference at
rtol 1e-3."
```

---

## Task 8: Decoder graph and mask output

**Files:**
- Modify: `scripts/includes/af_stemsplit.c`

**Interfaces:**
- Consumes: everything from Task 7.
- Produces:
  - `static struct ggml_tensor *ss_crop2d(struct ggml_context *g, struct ggml_tensor *x, int off, int n0, int n1)`
  - `static struct ggml_tensor *ss_decoder_block(struct ggml_context *g, struct ggml_tensor *x, struct ggml_tensor *skip, const SSLayer *l)`
  - `static int ss_infer(AVFilterContext *ctx, const float *mag, float *out[2])` — `mag` is `[F,T,C]` C-order, `out[i]` receives each instrument's estimated magnitude, same layout

- [ ] **Step 1: Write the failing test**

Same dump-and-compare as Task 7, without the layer filter:

```bash
docker run --rm -v "$PWD/out:/out" -v "/tmp/dump:/dump" spleeter-gguf compare.py \
    --fixtures /out/fixtures.json --dump /dump
```

Expected once implemented: `26/26 layers match`.

- [ ] **Step 2: Run it and confirm it fails**

Expected: encoder layers match, all 14 decoder and output layers are reported missing.

- [ ] **Step 3: Implement the decoder block**

```c
/* ggml_conv_transpose_2d_p0 does no padding, so a stride-2 kernel-5 transpose
 * of an N-wide input yields 2N+3. TensorFlow 'SAME' yields 2N. Drop 1 from the
 * front and 2 from the back to land on the same grid the encoder used.
 */
static struct ggml_tensor *ss_decoder_block(struct ggml_context *g,
                                            struct ggml_tensor *x,
                                            struct ggml_tensor *skip,
                                            const SSLayer *l)
{
    x = ggml_conv_transpose_2d_p0(g, l->w, x, /*stride*/ 2);
    x = ss_crop2d(g, x, /*off*/ 1, x->ne[0] - 3, x->ne[1] - 3);
    x = ggml_add(g, x, ggml_reshape_4d(g, l->b, 1, 1, l->b->ne[0], 1));
    x = ggml_relu(g, x);          /* decoder activates BEFORE normalising */
    x = ss_bn(g, x, l);
    return skip ? ggml_concat(g, skip, x, /*dim*/ 2) : x;
}
```

`ss_crop2d` is a `ggml_view_4d` at the byte offset `off * nb[0] + off * nb[1]` followed by `ggml_cont`, since the downstream convolution needs a contiguous tensor.

Note the concat order: Spleeter concatenates `[encoder_output, decoder_output]`, so the skip tensor comes first. Reversing it silently reorders the input channels of the next convolution.

- [ ] **Step 4: Implement the output layer and `ss_infer`**

```c
x = ggml_conv_2d(g, net->out.w, x, /*s0*/ 1, /*s1*/ 1, /*p0*/ 3, /*p1*/ 3, /*d0*/ 2, /*d1*/ 2);
x = ggml_add(g, x, ggml_reshape_4d(g, net->out.b, 1, 1, 2, 1));
x = ggml_sigmoid(g, x);
x = ggml_mul(g, x, input_spectrogram);   /* the estimated magnitude */
```

Kernel 4 with dilation 2 has an effective width of 7, so TensorFlow SAME pads 3 on each side — symmetric, and ggml handles it natively. Set the thread count from `ff_filter_get_nb_threads(ctx)` when `threads` is 0.

- [ ] **Step 5: Assert decoder shapes**

Expected `[F,T,C]` after each block, before concatenation: `[32,16,256]`, `[64,32,128]`, `[128,64,64]`, `[256,128,32]`, `[512,256,16]`, `[1024,512,1]`; the output layer gives `[1024,512,2]`.

- [ ] **Step 6: Run the full parity comparison**

Expected: `26/26 layers match`.

- [ ] **Step 7: Commit**

```bash
git add scripts/includes/af_stemsplit.c
git commit -m "feat(stemsplit): add the decoder graph and mask output

Six transposed-convolution blocks with skip concatenation and the
dilated sigmoid output layer. The transpose emits 2N+3 columns where
TensorFlow SAME emits 2N, so each block crops 1 from the front and 2
from the back to stay on the encoder's grid.

The decoder activates before normalising, unlike the encoder, which is
why the converter exports every BatchNorm as a standalone affine.

All 26 layers of both networks now match the Python reference."
```

---

## Task 9: Masks, high-band extension, segmentation, and outputs

Turns two magnitude estimates into audible stems. This is the first task whose output a human can listen to.

**Files:**
- Modify: `scripts/includes/af_stemsplit.c`

**Interfaces:**
- Consumes: `ss_infer` from Task 8, `ss_analyze`/`ss_synthesize` from Task 5.
- Produces:
  - `static void ss_build_masks(StemSplitContext *s, const float *est[2], float *mask[2])` — `[F,T,C]` in, `[SS_BINS,T,C]` out, extension applied
  - `static int ss_process_segment(AVFilterContext *ctx)`
  - `static int ss_config_output(AVFilterContext *ctx)` — appends output pads
  - Removes the temporary `passthrough_dsp` option from Task 5.

- [ ] **Step 1: Write the failing test — synthetic separation**

A mix whose parts are unambiguous by frequency: a 60 Hz sine for bass, band-limited noise for percussion, and a 900 Hz tone standing in for voice. Assert that the two outputs are not identical to each other and that both are shorter-or-equal in energy than the mixture — a smoke-level guard that masks are actually being applied.

```bash
docker run --rm -v "$PWD/out:/models" ffstem-vol:latest sh -c '
  cd /build/ffmpeg
  ./ffmpeg -y -f lavfi -i "sine=frequency=60:duration=15:sample_rate=44100" \
           -f lavfi -i "sine=frequency=900:duration=15:sample_rate=44100" \
           -filter_complex "[0:a][1:a]amix=inputs=2,aformat=channel_layouts=stereo" /tmp/mix.wav
  ./ffmpeg -y -i /tmp/mix.wav -filter_complex \
     "[0:a]stemsplit=model=/models/spleeter-2stems-f16.gguf:stem=all[v][a]" \
     -map "[v]" /tmp/voc.wav -map "[a]" /tmp/acc.wav
  cmp -s /tmp/voc.wav /tmp/acc.wav && echo "FAIL: stems identical" || echo "OK: stems differ"
  ./ffmpeg -i /tmp/voc.wav -af volumedetect -f null - 2>&1 | grep max_volume'
```

Expected once implemented: `OK: stems differ` and a finite `max_volume`.

- [ ] **Step 2: Run it and confirm it fails**

Expected: `Filter stemsplit has 1 outputs but 2 were specified`.

- [ ] **Step 3: Implement the masks**

```c
/* mask_i = (est_i^2 + EPSILON/N) / (sum_j est_j^2 + EPSILON) */
static void ss_build_masks(StemSplitContext *s, const float *est[2], float *mask[2])
{
    const int n = s->nb_instruments;
    for (int c = 0; c < SS_CHANNELS; c++)
        for (int t = 0; t < SS_T; t++)
            for (int f = 0; f < SS_F; f++) {
                const size_t k = ((size_t) c * SS_T + t) * SS_F + f;
                float sum = SS_EPSILON;
                for (int i = 0; i < n; i++)
                    sum += est[i][k] * est[i][k];
                for (int i = 0; i < n; i++)
                    mask[i][ss_idx(c, t, f)] =
                        (est[i][k] * est[i][k] + SS_EPSILON / n) / sum;
            }
}
```

Then extend bins `SS_F .. SS_BINS-1` per `highband`:
- `SS_HB_ZEROS` — zero in every stem.
- `SS_HB_AVERAGE` — the mean of that frame's mask over the modelled bins, tiled.
- `SS_HB_PASSTHROUGH` (default) — `1.0` for the **last** instrument in the model's list, `0.0` for the others.

- [ ] **Step 4: Implement segmentation and output**

Maintain an input ring buffer that yields `SS_T` frames at a time. Feed `SS_FRAME_LENGTH` zeros before the first sample; discard the first `SS_FRAME_LENGTH` output samples. At EOF, zero-pad the last partial segment and truncate the output to the true sample count. When `overlap` is non-zero, advance by `SS_T` frames minus the overlap and equal-power crossfade the overlapping output region.

Multiply the stored complex STFT by each mask and call `ss_synthesize` per stem, per channel.

- [ ] **Step 5: Implement the output pads**

Add `AVFILTER_FLAG_DYNAMIC_OUTPUTS` to `.p.flags` and in `init()`:

```c
if (s->stem == SS_STEM_ALL) {
    for (int i = 0; i < 2; i++) {
        AVFilterPad pad = { .name = i ? "accompaniment" : "vocals", .type = AVMEDIA_TYPE_AUDIO };
        int ret = ff_append_outpad(ctx, &pad);
        if (ret < 0)
            return ret;
    }
} else {
    AVFilterPad pad = { .name = "default", .type = AVMEDIA_TYPE_AUDIO };
    int ret = ff_append_outpad(ctx, &pad);
    if (ret < 0)
        return ret;
}
```

Remove `FILTER_OUTPUTS(...)` from the filter definition, since the pads are now dynamic. Extend `activate()` to forward status and frames across both outputs.

- [ ] **Step 6: Run the synthetic test, then listen**

Expected: `OK: stems differ`. Then a real check on actual music:

```bash
docker run --rm -v "$PWD/out:/models" -v "$PWD/sample:/audio" ffstem-vol:latest sh -c '
  cd /build/ffmpeg
  ./ffmpeg -y -i /audio/song.flac -af "stemsplit=model=/models/spleeter-2stems-f16.gguf:stem=accompaniment" \
     /audio/karaoke.flac'
```

Expected: the vocal is substantially attenuated and the result has no gross clicks at 11.9 s boundaries. This is the one subjective checkpoint in the plan; it comes *after* numeric parity, never instead of it.

- [ ] **Step 7: Verify `highband=zeros` reproduces upstream**

```bash
docker run --rm -v "$PWD/out:/models" -v "$PWD/sample:/audio" ffstem-vol:latest sh -c '
  cd /build/ffmpeg
  ./ffmpeg -y -i /audio/song.flac \
    -af "stemsplit=model=/models/spleeter-2stems-f16.gguf:stem=accompaniment:highband=zeros,showspectrumpic=s=640x480" \
    /audio/spec_zeros.png'
```

Expected: the spectrogram is visibly empty above 11 kHz. Repeat with the default `highband=passthrough` and confirm the high band is present. This is the visual proof that both modes do what the spec says.

- [ ] **Step 8: Commit**

```bash
git add scripts/includes/af_stemsplit.c
git commit -m "feat(stemsplit): add ratio masking, high-band handling and stem outputs

Converts the two magnitude estimates into ratio masks with separation
exponent 2, applies them to the original complex STFT so the mixture
phase is reused, and resynthesises each stem.

Above 11 kHz the model predicts nothing. The default now routes that
band into the accompaniment stem instead of zeroing it, which keeps
cymbals and air in a karaoke track; highband=zeros restores upstream
Spleeter's behaviour for comparison.

stem=all exposes one output pad per instrument; stem=<name> exposes a
single pad so the common case works with plain -af."
```

---

## Task 10: Cross-platform build, documentation, release

**Files:**
- Modify: `README.md`
- Verify: all seven platform builds

**Interfaces:**
- Consumes: the finished filter.
- Produces: no new code. A verified build matrix, user documentation, and the model published as a release asset.

- [ ] **Step 1: Build every target**

```bash
docker compose build ffmpeg-linux-x86_64 ffmpeg-linux-aarch64 \
                     ffmpeg-windows-x86_64 ffmpeg-windows-aarch64 \
                     ffmpeg-darwin-x86_64 ffmpeg-darwin-arm64 \
                     ffmpeg-freebsd-x86_64
```

Expected: all seven succeed. `freebsd-x86_64` and `windows-aarch64` have no BLAS and use ggml's own kernels — the same situation whisper is already in on those targets, so a link error mentioning `openblas` there means the filter picked up a dependency it should not have.

- [ ] **Step 2: Run the test suite on the host platform's artifact**

```bash
tests/tests.sh ./output/extracted --platform linux-x86_64 --json ./output/report.json
```

Expected: the `STEMSPLIT` test passes and nothing else regressed.

- [ ] **Step 3: Document the filter in the README**

Add to the custom-features table, next to the `keydetect` and `beatdetect` rows near `README.md:179`:

```markdown
| **`stemsplit`** | Audio filter | Music source separation into vocal and accompaniment stems (Spleeter 2stems on ggml) |
```

Below the table, add a short usage section with both example command lines from the spec's section 8, the option table, the model download link, a note that the filter needs about 12 seconds of lookahead and is intended for offline processing, the MIT licence and JOSS citation for Spleeter, and upstream's advisory about rights to copyrighted material.

- [ ] **Step 4: Verify the documented commands actually run**

Copy each command out of the README and run it against the built binary. A documented command that does not work is worse than no documentation.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(stemsplit): document the source separation filter

Adds stemsplit to the custom-features table with both usage forms, the
full option table, the model download, and the Spleeter licence and
citation. Notes the ~12 second lookahead so callers know it is an
offline filter."
```

- [ ] **Step 6: Publish the model as a release asset**

Upload `spleeter-2stems-f16.gguf` to a GitHub release on this repository and confirm the README's download link resolves. **Ask before doing this** — it is an outward-facing publish and needs explicit sign-off.

---

## Self-Review

**Spec coverage.** Section 4 constants → Global Constraints and Tasks 5, 8, 9. Section 5.1 runtime → Task 1. Section 6.1 converter → Tasks 2 and 3. Section 6.2 filter → Tasks 4, 5, 9. Section 6.3 ggml graph → Tasks 6, 7, 8. Section 6.4 build → Task 4. Section 7 high band → Task 9 steps 3 and 7. Section 8 interface → Tasks 4 and 9. Section 9.1 round trip → Task 5. Section 9.2 parity → Tasks 3, 7, 8. Section 9.3 synthetic → Task 9. Section 9.4 smoke → Task 4. Section 9.5 harness → Task 1. Section 10 error handling → Task 6. Section 11 licensing → Tasks 2 and 10. Section 12 documentation → Tasks 2 and 10. No gaps.

**Type consistency.** `SSLayer`/`SSNet` are defined in Task 6 and used unchanged in Tasks 7 and 8. `ss_pad_tf` is defined in Task 7 and used in Task 7 only; `ss_crop2d` in Task 8 only. `ss_bn` is defined in Task 7 and reused in Task 8. `ss_analyze`/`ss_synthesize` are defined in Task 5 and used in Task 9. `ss_infer` is produced by Task 8 and consumed by Task 9. The `dump=` option is declared in Task 4's option table and first implemented in Task 7. `passthrough_dsp` is introduced in Task 5 and explicitly removed in Task 9.

**Known open item, deliberately left to execution.** Task 1 decides the model container format, and Task 2's writer depends on that answer. This is sequenced rather than resolved because it cannot be answered without building the image. Both branches are fully specified.
