#!/usr/bin/env python3
"""Rebuild Spleeter's 2stems U-Net in Keras and dump per-layer reference
activations for a fixed synthetic input, so Tasks 7/8's ggml graph has
something to be checked against.

Produces:
  - `fixtures.json` (committed): 26 entries (`<inst>.<layer>`), each with
    shape, summary statistics and 64 fixed sample points.
  - `<raw-dir>/<inst>.<layer>.f32` (not committed): full activations, one
    file per layer, `[C][T][F]` byte order (F fastest) -- ggml's native
    memory layout for `ne = [F, T, C, 1]`.
  - `<raw-dir>/../fixtures_input.f32` (not committed) -- **interface
    contract, not just a byproduct**: the fixed input magnitude
    spectrogram every fixture in `fixtures.json` was computed from.
    Path: the parent directory of `--raw-dir` (`os.path.dirname` of it),
    filename `fixtures_input.f32`. Shape `[C=2][T=512][F=1024]` float32,
    same `[C][T][F]` byte order as the layer dumps above (F fastest). A
    later task injects this file as the network's input directly,
    bypassing the STFT (a planned `debug_input=<path>` filter option),
    so per-layer parity tests the network in isolation -- feeding real
    `anoisesrc` audio through the STFT instead produces a different
    spectrogram and cannot be compared against these fixtures.

Encoder taps: `conv1..conv6` are each the RAW, pre-BatchNorm,
pre-activation Conv2D output -- for every one of the six, not only
`conv6`. Every skip connection in the real graph concatenates the raw
conv (`merge1 = [conv5, drop1]` ... `merge5 = [conv1, batch11]`), never
the activated tensor (`rel1`..`rel5`) that feeds the next encoder stage.
A ggml implementation must keep each encoder layer's pre-BN tensor alive
and expose it separately from the BN+LeakyReLU'd tensor used internally
to compute the next stage -- getting this backwards is a real
separation-quality bug in the shipped filter, not just a fixture
mismatch. See `build_unet()` below for the full derivation, including
the conv6 bottleneck's extra wrinkle (its BN/activation are pure dead
code, computed but never consumed anywhere).

Usage:
    python dump_reference.py --checkpoint /work/2stems \
        --fixtures /out/fixtures.json --raw-dir /out/ref

Architecture reference: spleeter/model/functions/unet.py (Deezer, MIT
licensed; https://github.com/deezer/spleeter). Fetched and read directly
rather than assumed -- see `build_unet()` for the encoder-tap rule.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

from convert import BN_EPS, INSTRUMENTS, LAYER_ORDER, build_mapping

SEED = 20260819
N_T = 512
N_F = 1024
N_C = 2
N_SAMPLES = 64

# Encoder/decoder filter counts, read straight off convert.py's EXPECTED
# shape table so this can't drift from what the converter (and the GGUF
# tensor names it emits) actually produce.
ENC_FILTERS = [16, 32, 64, 128, 256, 512]
DEC_FILTERS = [256, 128, 64, 32, 16, 1]


def build_unet(input_tensor):
    """Reproduce `apply_unet()` from spleeter/model/functions/unet.py.

    Encoder block: Conv2D(5,5,stride2,same) -> BatchNorm -> LeakyReLU(0.2).
    Decoder block: Conv2DTranspose(5,5,stride2,same) -> ReLU -> BatchNorm
    -> concat(skip, x). Dropout(0.5) sits between BatchNorm and concat on
    up1-up3 in the source; it is a no-op at inference (training=False) and
    is omitted here rather than modelled as an identity layer.

    Skip connections use the RAW conv output, every time, not just at the
    bottleneck. Upstream:
        merge1 = Concatenate(axis=-1)([conv5, drop1])
        merge2 = Concatenate(axis=-1)([conv4, drop2])
        merge3 = Concatenate(axis=-1)([conv3, drop3])
        merge4 = Concatenate(axis=-1)([conv2, batch10])
        merge5 = Concatenate(axis=-1)([conv1, batch11])
    Every one of these concatenates `conv1..conv5` -- the pre-BatchNorm,
    pre-activation Conv2D output -- never `rel1..rel5` (the BN+LeakyReLU'd
    tensor that feeds the *next* encoder stage). So each encoder layer has
    two distinct consumers reading two distinct tensors: `rel_n` feeds
    `conv_{n+1}` going down the encoder, and the separate raw `conv_n`
    feeds the matching decoder block's concat coming back up. `taps["conv1"
    .."conv6"]` below are therefore all the raw tensor -- this is a general
    rule across every encoder layer, verified against the literal source
    above, not an assumption or a conv6-only special case.

    The conv6 bottleneck applies the same rule one step further, plus an
    extra wrinkle: upstream computes
        conv6 = Conv2D(...)(rel5)
        batch6 = BatchNormalization(...)(conv6)
        _ = conv_activation_layer(batch6)          # <- discarded
        up1 = Conv2DTranspose(...)((conv6))         # <- fed from conv6, RAW
    batch6 and the LeakyReLU that would follow it are computed (so the
    checkpoint carries trained gamma/beta/moving_mean/moving_variance for
    it) but their output is never used anywhere -- not as a skip source
    like conv1..conv5's raw tensors, not as anything. `taps["conv6"]` is
    the same raw tensor as conv1..conv5's taps; it just also happens to be
    conv6's *only* consumer, since there is no `rel6` feeding a seventh
    encoder stage. Getting either the general rule or this extra wrinkle
    backwards would make the corresponding downstream fixture(s) wrong.

    Returns (masked_output, taps) where taps maps the 13 layer ids this
    task must capture ("conv1".."conv6", "up1".."up6", "out") to tensors.
    """
    taps = {}
    encoder_raw = []  # conv1..conv6 raw outputs, pre-BN (used for skips + bottleneck)

    x = input_tensor
    for n in range(6):
        conv = layers.Conv2D(
            ENC_FILTERS[n], (5, 5), strides=(2, 2), padding="same", name=f"conv{n + 1}"
        )(x)
        taps[f"conv{n + 1}"] = conv
        encoder_raw.append(conv)
        bn = layers.BatchNormalization(axis=-1, epsilon=BN_EPS, name=f"conv{n + 1}_bn")(
            conv, training=False
        )
        act = layers.LeakyReLU(0.2)(bn)
        if n < 5:
            x = act
        # n == 5 (conv6): `act`/`bn` are dead ends, matching upstream's `_ = ...`.

    skip_for = [4, 3, 2, 1, 0]  # up1..up5 concat with conv5..conv1 (encoder_raw index)
    x = encoder_raw[5]  # up1 consumes raw conv6, not batch6/its activation
    for n in range(6):
        x = layers.Conv2DTranspose(
            DEC_FILTERS[n], (5, 5), strides=(2, 2), padding="same", name=f"up{n + 1}"
        )(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization(axis=-1, epsilon=BN_EPS, name=f"up{n + 1}_bn")(
            x, training=False
        )
        taps[f"up{n + 1}"] = x
        if n < 5:
            x = layers.Concatenate(axis=-1)([encoder_raw[skip_for[n]], x])

    out = layers.Conv2D(
        2, (4, 4), dilation_rate=(2, 2), padding="same", activation="sigmoid", name="out"
    )(x)
    taps["out"] = out
    masked = layers.Multiply()([out, input_tensor])

    assert list(taps.keys()) == LAYER_ORDER, (
        f"tap order {list(taps.keys())} does not match convert.py's LAYER_ORDER {LAYER_ORDER}"
    )
    return masked, taps


def load_weights(reader, mapping, model):
    """Set every Conv2D/Conv2DTranspose/BatchNormalization layer's weights
    from the checkpoint, using build_mapping()'s resolved TF variable names.
    Weights are pulled TF-native (no ggml axis swap, no fp16 cast) since
    this Keras model must run the real TF numerics, not the runtime's
    packed layout."""
    for layer_id, entry in mapping.items():
        _, layer_name = layer_id.split(".", 1)
        conv_name = entry["conv"]
        kernel = reader.get_tensor(f"{conv_name}/kernel").astype(np.float32)
        bias = reader.get_tensor(f"{conv_name}/bias").astype(np.float32)
        keras_layer = model.get_layer(layer_name)
        keras_layer.set_weights([kernel, bias])

        # conv6's BatchNorm/LeakyReLU are dead code in the real graph (see
        # build_unet's docstring): up1 consumes raw conv6, so that branch
        # is unreachable from `outputs` and Keras prunes it from the
        # traced model entirely. Its weights would have zero effect on
        # anything captured here, so there is nothing to load.
        if layer_name == "conv6":
            continue

        if entry["bn"] is not None:
            gamma = reader.get_tensor(f"{entry['bn']}/gamma").astype(np.float32)
            beta = reader.get_tensor(f"{entry['bn']}/beta").astype(np.float32)
            mean = reader.get_tensor(f"{entry['bn']}/moving_mean").astype(np.float32)
            var = reader.get_tensor(f"{entry['bn']}/moving_variance").astype(np.float32)
            bn_layer = model.get_layer(f"{layer_name}_bn")
            bn_layer.set_weights([gamma, beta, mean, var])


def dump_layer(raw_dir, layer_id, act_tfc):
    """act_tfc: (T, F, C) numpy array (Keras' native axis order for one
    segment). Writes the raw fixture as ggml's [C][T][F] byte order (F
    fastest) and returns the fixtures.json entry for this layer.

    Sample positions are drawn from a *fresh* `default_rng(SEED)` here,
    once per layer, rather than one continuing stream shared across all
    26 layers. That makes each layer's 64 positions depend only on that
    layer's own shape, not on the order layers happen to be processed in
    -- reproducible layer-by-layer in isolation, not only as a full run."""
    t, f, c = act_tfc.shape
    act_ctf = np.ascontiguousarray(np.transpose(act_tfc, (2, 0, 1)))  # (C, T, F)
    act_ctf.tofile(os.path.join(raw_dir, f"{layer_id}.f32"))

    rng = np.random.default_rng(SEED)
    fs = rng.integers(0, f, size=N_SAMPLES)
    ts = rng.integers(0, t, size=N_SAMPLES)
    cs = rng.integers(0, c, size=N_SAMPLES)
    samples = [
        [int(fi), int(ti), int(ci), float(act_tfc[ti, fi, ci])]
        for fi, ti, ci in zip(fs, ts, cs)
    ]

    return {
        "shape": [f, t, c],
        "mean": float(np.mean(act_tfc)),
        "std": float(np.std(act_tfc)),
        "min": float(np.min(act_tfc)),
        "max": float(np.max(act_tfc)),
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="directory containing the TF checkpoint")
    ap.add_argument("--fixtures", required=True, help="path to write fixtures.json to")
    ap.add_argument("--raw-dir", required=True, help="directory to write raw .f32 dumps to")
    args = ap.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)

    reader = tf.train.load_checkpoint(args.checkpoint)
    shape_map = reader.get_variable_to_shape_map()
    mapping = build_mapping(shape_map)

    mag_tfc = (np.random.default_rng(SEED).random((N_T, N_F, N_C)) * 4.0).astype(np.float32)
    input_dir = os.path.dirname(os.path.normpath(args.raw_dir))
    os.makedirs(input_dir, exist_ok=True)
    np.ascontiguousarray(np.transpose(mag_tfc, (2, 0, 1))).tofile(
        os.path.join(input_dir, "fixtures_input.f32")
    )

    fixtures = {}
    for inst in INSTRUMENTS:
        input_tensor = layers.Input(shape=(N_T, N_F, N_C), batch_size=1, name=f"{inst}_input")
        _, taps = build_unet(input_tensor)
        inst_mapping = {k: v for k, v in mapping.items() if k.startswith(f"{inst}.")}
        model = Model(inputs=input_tensor, outputs=list(taps.values()))
        load_weights(reader, inst_mapping, model)

        outputs = model(mag_tfc[np.newaxis, ...], training=False)
        for layer_name, act in zip(taps.keys(), outputs):
            act = np.asarray(act)[0]  # drop batch dim -> (T, F, C)
            layer_id = f"{inst}.{layer_name}"
            fixtures[layer_id] = dump_layer(args.raw_dir, layer_id, act)
            print(f"  {layer_id}: shape={fixtures[layer_id]['shape']}")

    with open(args.fixtures, "w") as fh:
        json.dump(fixtures, fh, separators=(",", ":"))

    size_kb = os.path.getsize(args.fixtures) / 1024
    print(f"wrote {len(fixtures)} layer fixtures")
    print(f"fixtures.json: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
