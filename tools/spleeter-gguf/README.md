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
three shape checks, calls `verify()`, and — only if that passes — writes the
GGUF file. Expected: `verify OK: 100 tensors`, then
`wrote /out/spleeter-2stems-f16.gguf`, roughly 39 MB.

Do not commit the output file (`out/` or wherever you point `--output`) —
copy it to wherever this repository's release pipeline picks up model
assets instead.

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
convert commands above. `convert.py`'s `EXPECTED` table and `build_mapping()`
shape checks will fail loudly if the checkpoint's variable names, shapes, or
instrument-split point ever change — treat any such failure as a real
architecture change to investigate, not something to route around.
