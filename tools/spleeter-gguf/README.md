# spleeter-gguf

Dockerised converter that turns Deezer's Spleeter `2stems` TensorFlow
checkpoint into `spleeter-2stems-f16.gguf`, the fp16 GGUF model file the
`stemsplit` FFmpeg audio filter loads at runtime.

**The `.gguf` output is a release asset. It is never committed to this
repository** — it is roughly 39 MB, far over the repo's 1 MB file-size
limit, and it is a build artifact regenerated from a checkpoint that is
itself downloaded at image-build time, not stored here.

## What this converts

Spleeter's `2stems` checkpoint contains two independent U-Nets, one per
instrument (`vocals`, `accompaniment`), each with 6 encoder `Conv2D` layers,
6 decoder `Conv2DTranspose` layers, a BatchNorm after every one of those 12
layers, and a final sigmoid output `Conv2D` with no BatchNorm — 100 tensors
total across both networks (50 each: 6 conv x 4 tensors + 6 up x 4 tensors +
2 for `out`).

The checkpoint's TensorFlow variable names carry **no** instrument prefix —
both networks share one flat Keras autonaming sequence
(`conv2d`, `conv2d_1`, ... `conv2d_13`; `conv2d_transpose`, ...
`conv2d_transpose_11`; `batch_normalization`, ... `batch_normalization_23`).
The two networks are distinguished only by *creation order*: Spleeter builds
one complete U-Net per entry of `instrument_list == ["vocals",
"accompaniment"]`. `convert.py` resolves this mapping from that construction
order, **then proves it** against three shape facts that are only true if
the split point is correct (`conv2d`/`conv2d_7` both take 2 input channels;
`conv2d_6`/`conv2d_13` are both the 4x4 1-to-2-channel output conv;
`conv2d_transpose_5`/`conv2d_transpose_11` both emit 1 channel) before
converting anything, and prints the full resolved mapping so a human can
eyeball it.

## Output format

- Tensor names: `<inst>.conv<N>.{weight,bias,bn_a,bn_b}` (N = 1..6),
  `<inst>.up<N>.{weight,bias,bn_a,bn_b}` (N = 1..6), `<inst>.out.{weight,bias}`
  — `<inst>` is `vocals` or `accompaniment`. No BatchNorm follows `out`.
- `*.weight` — **F16**, `[5, 5, IC, OC]` for `conv`, `[5, 5, OC, IC]` for `up`,
  `[4, 4, 1, 2]` for `out`. Stored with the kh/kw axes swapped relative to
  TensorFlow's `[kh, kw, ...]` layout, to match ggml's `[kw, kh, ...]`
  convention.
- `*.bias`, `*.bn_a`, `*.bn_b` — **F32**. `bn_a`/`bn_b` are BatchNorm reduced
  to a per-channel affine `y = a*x + b` (`a = gamma / sqrt(var + eps)`,
  `b = beta - mean * a`, `eps = 1e-3`, Keras's `BatchNormalization` default),
  so the runtime filter needs no BatchNorm special case, just one multiply
  and one add per channel after each conv.
- Metadata: `stemsplit.arch = "spleeter-unet-v1"`,
  `stemsplit.sample_rate = 44100`, `stemsplit.frame_length = 4096`,
  `stemsplit.frame_step = 1024`, `stemsplit.n_t = 512`, `stemsplit.n_f = 1024`,
  `stemsplit.separation_exponent = 2`,
  `stemsplit.instruments = ["vocals", "accompaniment"]`.

## Build

```bash
docker build -t spleeter-gguf tools/spleeter-gguf
```

This installs TensorFlow-CPU 2.16.2, NumPy 1.26.4 and the `gguf` Python
package (0.10.0), and downloads the MIT-licensed `2stems.tar.gz` checkpoint
(~73 MB) from the Spleeter v1.4.0 GitHub release into the image at
`/work/2stems`.

Note: `gguf==0.10.0`'s `gguf/vocab.py` unconditionally imports
`sentencepiece` even though that package declares only `numpy`, `pyyaml`
and `tqdm` as dependencies — without installing `sentencepiece` separately,
`import gguf` fails. The Dockerfile installs it explicitly; this is not a
Spleeter or `stemsplit` concern, just a packaging gap in that `gguf`
release.

## Verify (TDD red/green)

The converter's test is built into `convert.py` itself as `verify()`, so it
cannot drift from what the writer actually emits. Confirm it fails on an
empty tensor set first:

```bash
docker run --rm spleeter-gguf -c "import convert; convert.verify({})"
```

Expected: `AssertionError: missing tensors: [...]` listing all 100 names.

## Convert

```bash
docker run --rm -v "$PWD/out:/out" spleeter-gguf convert.py \
    --checkpoint /work/2stems --output /out/spleeter-2stems-f16.gguf
```

(On Windows Git Bash, prefix with `MSYS_NO_PATHCONV=1` — Git Bash otherwise
rewrites the leading `/work/...` and `/out/...` paths into Windows paths
before they reach the container.)

This prints the resolved instrument-to-checkpoint-variable mapping, runs the
three shape checks, calls `verify()` (tensor names and shapes) and then
`verify_dtypes()` (every `.weight` is float16, everything else is float32),
and — only if both pass — writes the GGUF file. Expected: `verify OK: 100
tensors`, then `verify_dtypes OK: 100 tensors`, then
`wrote /out/spleeter-2stems-f16.gguf`, roughly 39 MB.

Do not commit the output file (`out/` or wherever you point `--output`) —
copy it to wherever this repository's release pipeline picks up model
assets instead.

## Verify the shipped filter against the fixtures

`fixtures.json` (committed, ~55 KB) is the per-layer reference the `stemsplit`
filter is checked against: 26 entries, one per layer per instrument, each with
a shape, four summary statistics and 64 fixed sample points. It only guards
anything if somebody runs it, so here is the whole procedure against a built
`ffmpeg` binary. Any change to `af_stemsplit.c`'s numerics should be followed
by this, and a result other than `26/26 layers match` is a regression in that
change rather than a tolerance to loosen.

Five comments in `af_stemsplit.c` name this test as the thing that catches a
specific mistake: symmetric instead of asymmetric TensorFlow `SAME` padding,
symmetric instead of front-biased transpose cropping, `ggml_conv_2d` instead of
`ggml_conv_2d_direct`, and the two F32 kernel casts. Every one of those
produces correctly *shaped*, plausible-sounding output and raises no error.

Run all three steps from `tools/spleeter-gguf/`.

### 1. Regenerate `fixtures_input.f32`

`fixtures.json` on its own is not enough: the filter has to be fed the exact
input spectrogram the fixtures were computed from. That is `fixtures_input.f32`
— `[C=2][T=512][F=1024]` float32, F fastest, 4,194,304 bytes — which
`dump_reference.py` writes into the **parent** directory of `--raw-dir`. It is
a generated artifact and deliberately not committed (4 MB, over this repo's
1 MB file-size limit), so regenerate it before verifying:

```bash
mkdir -p out
docker run --rm -v "$PWD/out:/out" spleeter-gguf dump_reference.py \
    --checkpoint /work/2stems --fixtures /out/fixtures.json --raw-dir /out/ref
```

That writes `out/fixtures_input.f32`, the full reference activations in
`out/ref/<inst>.<layer>.f32` (also not committed) and `out/fixtures.json`. The
last of those should be identical to the committed `fixtures.json`; if it is
not, the fixtures and the checkpoint have drifted apart, and that is what to
investigate before anything else.

**`fixtures_input.f32` and `fixtures.json` are one unit.** The same run
produces both, describing the same forward pass. Regenerating the fixtures
without regenerating the input — or the reverse — makes the comparison
meaningless, because the filter would then be measured on a different
spectrogram than the reference statistics came from. Any change to the fixtures
must regenerate both.

### 2. Run the filter's parity path

`debug_input=` injects that spectrogram straight into the networks, bypassing
the STFT and the segment driver entirely, and `dump=` writes every layer tap to
disk. The audio input is irrelevant — the hook runs at filter init — but
`ffmpeg` still needs one, hence `anullsrc`. Use `stem=vocals` (or any single
stem): the default `stem=all` exposes two output pads and `-af` accepts only
one. The `dump` directory must already exist.

```bash
mkdir -p dump
M=spleeter-2stems-f16.gguf   # wherever you put the converted model
ffmpeg -hide_banner -y \
    -f lavfi -i "anullsrc=sample_rate=44100:channel_layout=stereo:d=0.1" \
    -af "stemsplit=model=$M:stem=vocals:debug_input=out/fixtures_input.f32:dump=dump" \
    -f null -
```

Both instruments' networks are dumped whatever `stem` says, because they share
one compute graph. Expect 30 files in `dump/`: the 26 fixture taps
(`<inst>.{conv1..conv6,up1..up6,out}.f32`) plus `<inst>.input.f32` and
`<inst>.est.f32`, two diagnostics `compare.py` ignores.

### 3. Compare

```bash
docker run --rm -v "$PWD:/w" spleeter-gguf compare.py \
    --fixtures /w/fixtures.json --dump /w/dump
```

Expected, and the only acceptable result:

```
26/26 layers match
```

Exit code 0. On failure `compare.py` exits 1 and prints one line per failing
layer, naming the statistic or sample point that disagreed — which localises a
regression to a layer. `--rtol` defaults to 1e-3 and `--layers <substring>`
narrows a run to, say, `conv` or `vocals` while bisecting; neither is a knob to
turn to make a failure go away. The tolerance is 1e-3 because the shipped
weights are F16 and F16 rounding alone reaches roughly that magnitude, so it is
already sized to the artifact rather than padded for comfort.

(On Windows Git Bash, prefix every `docker` command above with
`MSYS_NO_PATHCONV=1`, as with the convert step.)

## Licensing and attribution

The Spleeter source code and the `2stems` pretrained checkpoint are
released by Deezer under the **MIT License**
(https://github.com/deezer/spleeter/blob/master/LICENSE). This converter
downloads that checkpoint unmodified and reads its weights; it does not
redistribute Spleeter's source code.

If you use Spleeter (including via this converted model), please cite:

> Romain Hennequin, Anthony Khlif, Felix Voituret, Manuel Moussallam.
> **Spleeter: a fast and efficient music source separation tool with
> pre-trained models.** *Journal of Open Source Software*, 2020, 5(50), 2154.
> https://doi.org/10.21105/joss.02154

Deezer's own documentation advises that **users of Spleeter must own or
otherwise hold the legal rights to any copyrighted material they process**
with it (or with models derived from it, such as this GGUF conversion).
`stemsplit` inherits that same obligation: it is the responsibility of
whoever runs the filter to ensure they have the right to process the audio
they feed it.

## Regenerating this model

If this file needs to be regenerated (e.g. a future Spleeter release, or a
change to the tensor layout the filter expects): re-run the build and
convert commands above. `convert.py` runs two checks unconditionally on the
regen path, and either one failing means stop and investigate — not
something to route around:

- `verify()` and `build_mapping()`'s three shape checks fail loudly if the
  checkpoint's variable names, shapes, or instrument-split point ever
  change.
- `verify_dtypes()` fails loudly if any `.weight` tensor is not float16, or
  any `.bias`/`.bn_a`/`.bn_b` tensor is not float32 — e.g. if a future edit
  to `conv_weight()` drops the cast to F16, which `verify()` alone would
  not catch since it only checks shape.

Neither check inspects tensor *values* — only names, shapes, and dtypes.
Numerical correctness (that the BatchNorm reduction and axis swap actually
produce the right numbers) is Task 3's job (`dump_reference.py` /
`compare.py`), not this converter's.
