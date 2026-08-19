# Design: `stemsplit` — Spleeter music source separation as an FFmpeg audio filter

**Date:** 2026-08-19
**Branch:** `feat/spleeter-filter`
**Status:** Approved design, pending implementation plan

---

## 1. Summary

Add a native FFmpeg audio filter, `stemsplit`, that separates a music track into
`vocals` and `accompaniment` stems using the Deezer Spleeter `2stems` model.

The filter runs the network on **ggml**, which is already statically linked into
every one of this repo's seven platform binaries by way of `whisper.cpp`. No new
third-party dependency is introduced. The deliverable is one filter source file,
one build script, one offline model-conversion tool, and a dev harness.

---

## 2. Goals

- `2stems` separation (vocals / accompaniment) at 44.1 kHz stereo.
- Works on all seven build targets: linux x86_64/aarch64, windows x86_64/aarch64,
  darwin x86_64/arm64, freebsd x86_64.
- No new dependency in the static link; no growth of the shipped tarball.
- Output quality at least equal to upstream Spleeter, with an opt-out improvement
  to its band-limiting artifact.
- Numerically verifiable against the Python reference, layer by layer.

## 3. Non-goals

Explicitly **not** in this work, though none are precluded later:

- 4-stem and 5-stem models.
- GPU backends (CUDA / Metal / Vulkan).
- Real-time or low-latency operation. The algorithm requires ~11.9 s of lookahead.
- Multichannel Wiener filtering (Spleeter's optional `norbert` path). The released
  models are tuned for ratio masking, which is Spleeter's default.
- A generic multi-architecture model loader. The GGUF carries an architecture tag
  so a second architecture *can* be added later, but only one is implemented.
- Bundling model weights in the platform tarballs.

---

## 4. Background: the reference algorithm

All constants below are taken from `deezer/spleeter` at `master` and from
`spleeter/resources/2stems.json`. They are reproduced here because the
implementation must match them exactly.

### 4.1 Signal parameters

| Parameter | Value |
|---|---|
| `sample_rate` | 44100 |
| `n_channels` | 2 (stereo) |
| `frame_length` | 4096 |
| `frame_step` | 1024 |
| window | Hann, **periodic** |
| `T` (frames per segment) | 512 (approx. 11.888 s) |
| `F` (bins fed to network) | 1024 (0 to 11 025 Hz) |
| full bins in STFT | 2049 (`frame_length / 2 + 1`) |
| `separation_exponent` | 2 |
| `EPSILON` | 1e-10 |
| `WINDOW_COMPENSATION_FACTOR` | 2/3 |

### 4.2 Forward pipeline

1. Prepend **`frame_length` (4096) zeros** to the waveform, per channel.
2. STFT with the parameters above and `pad_end=True` (the tail is zero-padded so
   every sample is covered). Result is indexed `[time, bin, channel]`.
3. Zero-pad the time axis to a multiple of `T`, partition into `T`-frame segments,
   and keep the **first `F` bins** of the magnitude. Network input per segment is
   `[T=512, F=1024, C=2]`, raw magnitude with no normalisation.
4. Run one U-Net **per instrument**. Each emits a sigmoid mask which is multiplied
   by the network input to yield that instrument's estimated magnitude `S_i`.
5. Build ratio masks across instruments:

   ```
   output_sum = sum_j (S_j ^ 2) + EPSILON
   mask_i     = (S_i ^ 2 + EPSILON / N) / output_sum
   ```

6. Extend each mask from `F` bins to 2049 bins (section 7).
7. Multiply the **original complex mixture STFT** by the extended mask. The
   mixture's phase is reused unchanged.
8. Inverse STFT by weighted overlap-add — the Hann window is applied a second time
   on synthesis — and scale by `WINDOW_COMPENSATION_FACTOR` (2/3). For Hann
   periodic at 75 % overlap the analysis-synthesis window power sums to 1.5, so
   2/3 is the exact reconstruction gain.
9. Crop the result to `[frame_length, frame_length + original_length)`.

### 4.3 Network architecture (per instrument)

Encoder — six blocks, each `Conv2D 5x5 stride 2 SAME` then `BatchNorm` then
`LeakyReLU(0.2)`:

| Layer | Out channels | Out shape (T x F) |
|---|---|---|
| input | 2 | 512 x 1024 |
| conv1 | 16 | 256 x 512 |
| conv2 | 32 | 128 x 256 |
| conv3 | 64 | 64 x 128 |
| conv4 | 128 | 32 x 64 |
| conv5 | 256 | 16 x 32 |
| conv6 | 512 | 8 x 16 |

Decoder — six blocks, each `Conv2DTranspose 5x5 stride 2 SAME` then `ReLU` then
`BatchNorm` then `Dropout(0.5)` then concat with the matching encoder output:

| Layer | Out channels | Out shape | Concatenated with | Channels into next |
|---|---|---|---|---|
| up1 | 256 | 16 x 32 | conv5 | 512 |
| up2 | 128 | 32 x 64 | conv4 | 256 |
| up3 | 64 | 64 x 128 | conv3 | 128 |
| up4 | 32 | 128 x 256 | conv2 | 64 |
| up5 | 16 | 256 x 512 | conv1 | 32 |
| up6 | 1 | 512 x 1024 | — | 1 |

Output — `Conv2D 2ch, 4x4, dilation 2, SAME, sigmoid`, then multiplied elementwise
by the network input.

Approximately **9.82 M parameters per instrument**, about 20 MB at fp16. The
`2stems` configuration is two such networks, about **39 MB**.

**Note on activation ordering.** The encoder is `conv -> BN -> activation`, so its
BatchNorm folds backward into the convolution. The **decoder is
`convT -> activation -> BN`**, so its BatchNorm sits *after* a nonlinearity and
cannot fold backward. Rather than fold some layers and not others, the converter
reduces **every** BatchNorm to a per-channel affine `y = a*x + b` and the runtime
applies it as an explicit op. Cost is negligible; uniformity removes a whole class
of conversion bugs. Dropout is identity at inference and is dropped entirely.

### 4.4 Compute and memory budget

About 6.1 GMAC per 11.9 s segment per instrument, roughly **1 GFLOP per second of
audio per stem**. With a BLAS-backed gemm this is roughly **5–20x realtime per
core** — less work than whisper-base, which this repo already ships. Peak resident
memory is estimated below 200 MB: about 40 MB weights, 60 MB activations, 30 MB
im2col scratch.

---

## 5. Approach

### 5.1 Inference runtime — decision

| Option | New dependency | Static across 7 targets | Verdict |
|---|---|---|---|
| **ggml (already linked)** | none | yes | **chosen** |
| FFmpeg DNN framework via libtorch / OpenVINO / libtensorflow | ~2 GB | no | rejected |
| ONNX Runtime | large | no (freebsd, windows-aarch64) | rejected |
| Hand-written SIMD C | none | yes | rejected — re-implements ggml's per-arch gemm |

`scripts/48-whisper.sh` writes a `whisper.pc` exporting
`-lggml -lggml-base -lggml-cpu` (plus `-lggml-blas` on darwin and windows-x86_64).
The filter therefore declares `stemsplit_filter_deps="whisper"`. This is honest
about where libggml comes from and degrades gracefully — if whisper is ever
disabled, the filter is simply absent rather than breaking the build.

Upstream precedent supports this: FFmpeg's own `af_whisper` bypasses the DNN
framework and links whisper.cpp directly. FFmpeg's DNN filters are all video-only;
there is no audio DNN filter upstream and no source-separation filter of any kind.

### 5.2 Model choice — why Spleeter and not Demucs

Demucs v4 separates audibly better, but its **pretrained weights are CC-BY-NC 4.0**,
which rules them out for a commercial product. Spleeter's weights ship from an
MIT-licensed repository. That licensing gap is the deciding factor. The filter
therefore takes the model as a `model=` path rather than hardcoding it, so a
better permissively-licensed model later is a file swap, not a rewrite.

---

## 6. Components

Four units, each separately testable.

```
tools/spleeter-gguf/            offline, Dockerised, never shipped
    TF checkpoint -> per-channel BN affine -> fp16 GGUF -> GitHub release asset
                            |
                            v
scripts/includes/af_stemsplit.c
    segmenter -> STFT (av_tx) -> ggml U-Net x2 -> mask -> ISTFT/OLA -> outputs
                            |
scripts/60-stemsplit.sh     v         dev/ffstem-vol
    allfilters.c + Makefile + configure    fast rebuild harness
```

### 6.1 Unit 1 — model converter (`tools/spleeter-gguf/`)

A throwaway Docker image. TensorFlow is needed only to read the checkpoint; the
`spleeter` package itself is not required, since `tf.train.load_checkpoint`
suffices.

Responsibilities:

- Read the `2stems` checkpoint.
- Reduce each BatchNorm to per-channel `(a, b)` where
  `a = gamma / sqrt(var + eps)` and `b = beta - mean * a`.
- Transpose weights from TensorFlow's `[kh, kw, in, out]` (and
  `[kh, kw, out, in]` for transposed convolutions) to ggml's expected layout.
- Emit fp16 tensors into a GGUF file carrying metadata: architecture tag,
  version, sample rate, frame length, frame step, `T`, `F`,
  `separation_exponent`, and the ordered instrument list.
- **Also emit reference activations** for one fixed 11.9 s input segment — every
  one of the 13 layers per network — as the parity fixture for section 9.2.

Output artifact: `spleeter-2stems-f16.gguf`, about 39 MB, published as a GitHub
release asset. It is never committed to git and never bundled in the platform
tarballs.

### 6.2 Unit 2 — the filter (`scripts/includes/af_stemsplit.c`)

Follows the structure of `scripts/includes/af_whisper.c`: an `activate()`-driven
filter with an internal ring buffer.

- **Format negotiation** pins `AV_SAMPLE_FMT_FLTP`, stereo, 44 100 Hz. FFmpeg
  auto-inserts `aresample`, so arbitrary input still works.
- **Segmenter** accumulates 512 STFT frames (524 288 samples) plus the 4096-sample
  lead-in, emits a segment, and retains overlap state. At EOF the tail is
  zero-padded to a full segment and the output cropped back to the true length.
- **STFT / ISTFT** use `av_tx` (`libavutil/tx.h`) real-to-complex of size 4096 with
  a precomputed periodic Hann window, hop 1024, and 2/3 synthesis gain.
- **Inference** runs the ggml graph twice per segment, once per instrument.
- **Mask, extension, application** exactly as section 4.2 steps 5 to 7.
- **Output** is written to one or two output pads (section 8).

### 6.3 Unit 3 — the ggml graph

Built once at `init()` and reused; weights loaded from GGUF into a `ggml_context`,
allocation via `ggml_gallocr`, threads from `ff_filter_get_nb_threads()`.

The one genuinely delicate part is **TensorFlow `SAME` padding**, which for kernel
5 / stride 2 pads **1 before and 2 after** — asymmetric. `ggml_conv_2d` offers only
symmetric padding and `ggml_conv_transpose_2d_p0` offers none, so each layer type
is emulated explicitly:

| Layer | Emulation | Size check |
|---|---|---|
| `Conv2D 5x5 s2 SAME` | copy into a zeroed tensor padded 1 before / 2 after on both axes, then convolve with `p=0` | `(N+3-5)/2 + 1 = N/2` ok |
| `Conv2DTranspose 5x5 s2 SAME` | `ggml_conv_transpose_2d_p0` yields `2N+3`; **crop 1 from the front and 2 from the back** | `2N` ok |
| `Conv2D 4x4 dilation 2 SAME` | effective kernel 7, total pad 6, so symmetric `p=3`, native | `N` ok |

Symmetric padding is **not** an acceptable shortcut: with `p=2` the output shape
happens to come out right, but every window is shifted one input sample, which
produces plausible-sounding but incorrect audio. This is the same failure class as
the latent FFT-stride bug in `af_beatdetect.c`, and it is why section 9.2 exists.

Two ggml details to confirm during implementation, each with a stated fallback:

- `ggml_conv_2d` historically requires **F16 kernels** on the im2col path. If the
  pinned whisper.cpp exposes `ggml_conv_2d_direct` with F32 kernels, use it;
  otherwise keep weights F16, which is the intended format anyway.
- The GGUF reader (`gguf.h`) must be present in whisper.cpp's installed headers.
  If it is not installed, fall back to a small custom flat container — magic,
  version, tensor table — as `demucs.cpp` does. This decision is made in the first
  implementation step, before any other work depends on it.

#### 6.3.1 ggml capability probe results (probed 2026-08-19)

Both unknowns were settled against the pinned whisper.cpp 1.9.1 ggml, using a
throwaway Docker harness (`ffstem-vol:latest`, built from a scratchpad
Dockerfile that runs only `scripts/03-zlib.sh`, `scripts/04-fftw3.sh` and
`scripts/48-whisper.sh` against `nomercyentertainment/ffmpeg-base:latest`,
stopping before FFmpeg's own build so the configured source tree survives).

- **`HAS_GGUF_H` = yes.** `${PREFIX}/include/gguf.h` is installed by whisper's
  cmake install step, and `gguf_init_from_file()` compiles, links and runs
  (verified against a nonexistent path, which correctly returns an error
  rather than crashing). The GGUF reader is available — the plan proceeds
  with real GGUF, not the flat-container fallback.
  - Note for implementers: linking any translation unit that calls into
    `gguf_init_from_file` pulls in `ggml-quants.c`'s OpenMP-parallelized
    quantization initializers (`iq2xs_init_impl`, `iq3xs_init_impl`), which
    need `-fopenmp` (glibc: `-lgomp`) on the link line or you get spurious
    `undefined reference to GOMP_*` errors that look like "gguf.h is
    unusable" but are not. This is the same root cause as FFmpeg's own
    `./configure` needing `--pkg-config-flags=--static` to pick up
    `whisper.pc`'s `Libs.private: -lstdc++ -lm -fopenmp`
    (`scripts/48-whisper.sh:158-186`) — without it, FFmpeg's whisper
    pkg-config link-test fails the same way and configure misreports
    "whisper >= 1.7.5 not found".
- **`HAS_CONV2D_DIRECT` = yes.** `ggml_conv_2d_direct(ctx, a, b, s0, s1, p0,
  p1, d0, d1)` is declared in `ggml.h` and links cleanly with F32 kernels
  (kernel `a` is `[KW, KH, IC, OC]`, input `b` is `[W, H, C, N]`). Per section
  6.3, Unit 3 uses it instead of `ggml_conv_2d`'s im2col path, so weights do
  not need to be forced to F16 for the `Conv2D` layers.
- **`ggml_pad_ext` (Ruling 5 probe) = exists.** Signature:
  `ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)` — zero-pads
  each of the 4 dimensions independently on the left and right. This is a
  native per-side padding op, unlike `ggml_pad`/`ggml_pad_circular`
  (symmetric-only, `p0..p3`). It directly expresses the asymmetric
  TensorFlow `SAME` padding (1 before, 2 after) that section 6.3's `Conv2D
  5x5 s2 SAME` row currently emulates by hand via a zeroed intermediate
  tensor — a later task may use it to simplify that emulation, but this task
  makes no implementation change; it only confirms the op is available.

### 6.4 Unit 4 — build integration (`scripts/60-stemsplit.sh`)

Follows `scripts/57-keydetect.sh` verbatim in structure:

1. Copy `af_stemsplit.c` into `libavfilter/`.
2. `sed` the `extern const FFFilter ff_af_stemsplit;` declaration into
   `allfilters.c` after the last audio filter extern.
3. `sed` `OBJS-$(CONFIG_STEMSPLIT_FILTER) += af_stemsplit.o` into the Makefile.
4. Insert `stemsplit_filter_deps="whisper"` into `configure`.

Each step verifies itself and exits non-zero on failure, matching the existing
scripts. `--enable-filter=all` picks the filter up; no `add_enable` call is needed.

---

## 7. High-band extension — where we improve on upstream

`2stems.json` sets `"mask_extension": "zeros"`. Every bin above **11 025 Hz is
zeroed in every stem**. This is the origin of Spleeter's characteristic muffled
output, and it means the stems do not sum back to the mixture.

The `highband` option selects the behaviour:

| Value | Behaviour |
|---|---|
| `passthrough` | **default.** The untouched high band of the mixture is routed into exactly one stem and zeroed in all others. The receiving stem is the **last entry in the model's instrument list**, which for `2stems` is `accompaniment`. Cymbals and air survive in a karaoke track. |
| `zeros` | bit-compatible with upstream Spleeter. Used by the parity tests. |
| `average` | Spleeter's other documented mode: the mask's mean over the modelled bins is tiled across the high band. |

Defaulting to `passthrough` means the out-of-box result sounds better than the
Python tool; `highband=zeros` remains available for anyone comparing against it.

---

## 8. Filter interface

```bash
# karaoke: instrumental only
ffmpeg -i song.flac -af "stemsplit=model=2stems.gguf:stem=accompaniment" karaoke.flac

# both stems
ffmpeg -i song.flac -filter_complex \
  "[0:a]stemsplit=model=2stems.gguf:stem=all[v][a]" \
  -map "[v]" vocals.flac -map "[a]" music.flac
```

| Option | Type | Default | Description |
|---|---|---|---|
| `model` | string | *required* | path to the `.gguf` model |
| `stem` | enum | `all` | `vocals`, `accompaniment`, or `all` |
| `highband` | enum | `passthrough` | `passthrough`, `zeros`, or `average` |
| `overlap` | duration | `0` | crossfade between segments |
| `threads` | int | `0` | ggml CPU threads; 0 means the filter default |
| `dump` | string | `""` | **internal.** Directory to write per-layer tensors to, for the parity tests of section 9.2. Not documented in the README. |

`stem=all` uses `AVFILTER_FLAG_DYNAMIC_OUTPUTS`, appending output pads in `init()`
via `ff_append_outpad` — the `af_channelsplit` pattern. `stem=<name>` produces a
single output, which matters because the ffmpeg CLI treats an unconnected filter
output as an error; the single-stem form is what makes the common case usable with
plain `-af`.

`overlap` defaults to `0` for reference parity. Spleeter processes 11.9 s chunks
with no overlap and has audible seams; a non-zero value processes overlapping
segments and crossfades the output, at proportional cost.

---

## 9. Testing

### 9.1 STFT round-trip
Analysis followed by synthesis with no mask applied must reconstruct the input to
better than -90 dB. Catches window, hop, scaling and lead-in/crop errors before
any machine learning is involved.

### 9.2 Per-layer parity
For the fixed reference segment emitted by the converter, compare all 13 layer
outputs per network against the Python reference at `rtol=1e-3`. This is the
safety net for the padding-crop arithmetic in section 6.3 and must pass before any
subjective listening.

### 9.3 End-to-end separation
A synthetic mix — low sine (bass), band-limited noise (percussion), and a
vocal-band tone — with assertions on where energy lands. Deterministic, needs no
reference audio committed to the repo, and catches gross routing errors.

### 9.4 Smoke
`ffmpeg -filters | grep stemsplit` added to `tests/tests.sh` and `tests/tests.ps1`,
matching the coverage the other custom filters already have.

### 9.5 Dev harness
`dev/ffstem-vol`, mirroring the existing ffkey-vol / ffhls-vol harnesses: a
volume-mounted Docker image that rebuilds only the filter and reruns the tests, so
the padding work does not require a full cross-compile per iteration.

No large binaries enter git. Parity fixtures are per-layer checksums plus small
slices, not full tensors.

---

## 10. Error handling

| Condition | Behaviour |
|---|---|
| `model` not given | `AVERROR(EINVAL)` naming the `model` option |
| model file missing or unreadable | `AVERROR(EIO)` with the path and the release-asset URL |
| GGUF architecture tag unrecognised | `AVERROR(EINVAL)`, refuse to guess |
| GGUF tensor shape mismatch | `AVERROR(EINVAL)` naming the offending tensor |
| `stem` names a stem absent from the model | `AVERROR(EINVAL)` listing available stems |
| allocation failure | `AVERROR(ENOMEM)` |

ggml log output is bridged to `av_log` at matching severity, as `af_whisper.c`
does with `cb_log`.

---

## 11. Licensing and distribution

- Spleeter's code and released model checkpoints come from an MIT-licensed
  repository. The JOSS citation (Hennequin et al., 2020) is recorded in the
  converter's README and in the repo README's custom-features table.
- Upstream's own advisory is carried forward in documentation: users must hold
  the necessary rights to any copyrighted material they process.
- The model is a GitHub release asset on this repo, downloaded by the user or by
  the media server the same way whisper's ggml models are. Platform tarballs do
  not grow.

---

## 12. Documentation

- `README.md`: add `stemsplit` to the custom-features table alongside `keydetect`
  and `beatdetect`, with the two example command lines and the model download link.
- `tools/spleeter-gguf/README.md`: how to regenerate the model and the fixtures.

---

## 13. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Asymmetric SAME padding implemented wrong | **high** — output sounds plausible but is incorrect | per-layer parity tests (9.2) gate all other work |
| `gguf.h` not installed by whisper.cpp | medium — blocks model loading | decided in step 1; fallback is a flat container |
| `ggml_conv_2d` F16 kernel constraint | low | weights are fp16 by design |
| 11.9 s latency surprises a caller | low | documented as offline-only |
| Quality below expectations for some material | medium | Spleeter's known ceiling; `highband=passthrough` recovers part of it; model is swappable |
| freebsd / windows-aarch64 lack BLAS | low | ggml's own kernels are used, as they already are for whisper on those targets |
