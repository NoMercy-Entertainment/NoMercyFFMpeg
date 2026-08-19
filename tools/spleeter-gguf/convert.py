#!/usr/bin/env python3
"""Convert Spleeter's 2stems TensorFlow checkpoint into a fp16 GGUF file.

Produces `spleeter-2stems-f16.gguf`: two independent U-Nets ("vocals" and
"accompaniment"), 50 tensors each (6 encoder conv + 6 decoder up-conv, each
4 tensors: weight/bias/bn_a/bn_b, plus a 2-tensor output conv with no
BatchNorm) -- 100 tensors total.

Usage:
    # TDD red: prove verify() actually rejects a wrong/empty tensor set
    python -c "import convert; convert.verify({})"

    # Convert a checkpoint
    python convert.py --checkpoint /work/2stems --output /out/spleeter-2stems-f16.gguf
"""

import argparse

import numpy as np
import tensorflow as tf

BN_EPS = 1e-3  # Keras BatchNormalization default


# ---------------------------------------------------------------------------
# Step 2: the test. EXPECTED and verify() are written and run (against {})
# before the writer exists, so the assertion cannot silently drift from what
# the writer actually emits -- both live in this one file.
# ---------------------------------------------------------------------------

EXPECTED = {}
for inst in ("vocals", "accompaniment"):
    for n, (ic, oc) in enumerate([(2, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512)], 1):
        EXPECTED[f"{inst}.conv{n}.weight"] = (5, 5, ic, oc)
        EXPECTED[f"{inst}.conv{n}.bias"] = (oc,)
        EXPECTED[f"{inst}.conv{n}.bn_a"] = (oc,)
        EXPECTED[f"{inst}.conv{n}.bn_b"] = (oc,)
    for n, (ic, oc) in enumerate([(512, 256), (512, 128), (256, 64), (128, 32), (64, 16), (32, 1)], 1):
        EXPECTED[f"{inst}.up{n}.weight"] = (5, 5, oc, ic)
        EXPECTED[f"{inst}.up{n}.bias"] = (oc,)
        EXPECTED[f"{inst}.up{n}.bn_a"] = (oc,)
        EXPECTED[f"{inst}.up{n}.bn_b"] = (oc,)
    EXPECTED[f"{inst}.out.weight"] = (4, 4, 1, 2)
    EXPECTED[f"{inst}.out.bias"] = (2,)


def verify(tensors):
    missing = sorted(set(EXPECTED) - set(tensors))
    extra = sorted(set(tensors) - set(EXPECTED))
    assert not missing, f"missing tensors: {missing}"
    assert not extra, f"unexpected tensors: {extra}"
    for name, shape in EXPECTED.items():
        got = tuple(tensors[name].shape)
        assert got == shape, f"{name}: expected {shape}, got {got}"
    print(f"verify OK: {len(EXPECTED)} tensors")


# ---------------------------------------------------------------------------
# Step 4: the implementation.
# ---------------------------------------------------------------------------

def bn_affine(reader, prefix):
    """Reduce a BatchNorm to per-channel y = a*x + b."""
    gamma = reader.get_tensor(f"{prefix}/gamma")
    beta = reader.get_tensor(f"{prefix}/beta")
    mean = reader.get_tensor(f"{prefix}/moving_mean")
    var = reader.get_tensor(f"{prefix}/moving_variance")
    a = gamma / np.sqrt(var + BN_EPS)
    b = beta - mean * a
    return a.astype(np.float32), b.astype(np.float32)


def conv_weight(reader, name, transposed=False):
    """TF [kh,kw,ic,oc] (or [kh,kw,oc,ic] for a Conv2DTranspose kernel) ->
    ggml's [kw,kh,...] layout, cast to F16.

    `transposed` is accepted only for call-site documentation: both layouts
    need exactly the same kh/kw axis swap here, since axes 2 and 3 (whatever
    they mean for that kernel type) are left alone and TF's convention
    already puts them where GGUF/EXPECTED wants them.
    """
    w = reader.get_tensor(name)
    w = np.transpose(w, (1, 0, 2, 3))  # swap kh/kw only
    return np.ascontiguousarray(w).astype(np.float16)


# ---------------------------------------------------------------------------
# Layer-index mapping.
#
# The checkpoint's variable names carry NO instrument prefix (flat Keras
# autonaming: conv2d, conv2d_1, ... conv2d_13; conv2d_transpose, ... _11;
# batch_normalization, ... _23). The two instruments are distinguished only
# by creation order: spleeter builds one full U-Net per entry of
# instrument_list == ["vocals", "accompaniment"], each in the order
# conv1..conv6 (+BN each), up1..up6 (+BN each), then the output conv.
#
# That ordering is *derived from upstream source*, not read from the
# checkpoint, so it is proven below against the checkpoint's actual shapes
# before anything is converted. If any of the three checks fails, this
# raises immediately rather than silently mis-wiring the model.
# ---------------------------------------------------------------------------

INSTRUMENTS = ("vocals", "accompaniment")
LAYER_ORDER = [f"conv{n}" for n in range(1, 7)] + [f"up{n}" for n in range(1, 7)] + ["out"]


def tf_name(base, idx):
    """Keras autonaming: first instance is unsuffixed, then _1, _2, ..."""
    return base if idx == 0 else f"{base}_{idx}"


def build_mapping(shape_map):
    """Resolve the TF variable-name prefix for every logical layer of every
    instrument, proving the split point against real tensor shapes first."""

    def kshape(name):
        return tuple(shape_map[f"{name}/kernel"])

    # Check 1: only a network's conv1 takes 2 input channels.
    conv0 = kshape("conv2d")
    conv7 = kshape("conv2d_7")
    assert conv0 == (5, 5, 2, 16), f"conv2d/kernel: expected (5,5,2,16), got {conv0}"
    assert conv7 == (5, 5, 2, 16), f"conv2d_7/kernel: expected (5,5,2,16), got {conv7}"

    # Check 2: only the output conv is 4x4 with 1 input channel, 2 outputs.
    out_a = kshape("conv2d_6")
    out_b = kshape("conv2d_13")
    assert out_a == (4, 4, 1, 2), f"conv2d_6/kernel: expected (4,4,1,2), got {out_a}"
    assert out_b == (4, 4, 1, 2), f"conv2d_13/kernel: expected (4,4,1,2), got {out_b}"

    # Check 3: only up6 emits 1 channel.
    up6_a = kshape("conv2d_transpose_5")
    up6_b = kshape("conv2d_transpose_11")
    assert up6_a == (5, 5, 1, 32), f"conv2d_transpose_5/kernel: expected (5,5,1,32), got {up6_a}"
    assert up6_b == (5, 5, 1, 32), f"conv2d_transpose_11/kernel: expected (5,5,1,32), got {up6_b}"

    mapping = {}
    for inst_idx, inst in enumerate(INSTRUMENTS):
        conv_off = inst_idx * 7      # conv2d..: vocals 0..6, accompaniment 7..13
        up_off = inst_idx * 6        # conv2d_transpose..: vocals 0..5, accompaniment 6..11
        bn_conv_off = inst_idx * 12  # batch_normalization..: vocals 0..5, accompaniment 12..17
        bn_up_off = inst_idx * 12 + 6  # vocals 6..11, accompaniment 18..23

        for n in range(1, 7):
            mapping[f"{inst}.conv{n}"] = {
                "conv": tf_name("conv2d", conv_off + n - 1),
                "bn": tf_name("batch_normalization", bn_conv_off + n - 1),
                "transposed": False,
            }
            mapping[f"{inst}.up{n}"] = {
                "conv": tf_name("conv2d_transpose", up_off + n - 1),
                "bn": tf_name("batch_normalization", bn_up_off + n - 1),
                "transposed": True,
            }
        mapping[f"{inst}.out"] = {
            "conv": tf_name("conv2d", conv_off + 6),
            "bn": None,
            "transposed": False,
        }
    return mapping


def print_mapping(mapping):
    print("Resolved layer mapping (our tensor name -> TF checkpoint variable prefix):")
    for inst in INSTRUMENTS:
        print(f"  [{inst}]")
        for layer in LAYER_ORDER:
            entry = mapping[f"{inst}.{layer}"]
            bn = entry["bn"] or "(none)"
            print(f"    {layer:8s} conv={entry['conv']:24s} bn={bn}")


def gather_tensors(reader):
    shape_map = reader.get_variable_to_shape_map()
    mapping = build_mapping(shape_map)
    print_mapping(mapping)

    tensors = {}
    for key, entry in mapping.items():
        conv_name = entry["conv"]
        tensors[f"{key}.weight"] = conv_weight(reader, f"{conv_name}/kernel", entry["transposed"])
        tensors[f"{key}.bias"] = reader.get_tensor(f"{conv_name}/bias").astype(np.float32)
        if entry["bn"] is not None:
            a, b = bn_affine(reader, entry["bn"])
            tensors[f"{key}.bn_a"] = a
            tensors[f"{key}.bn_b"] = b
    return tensors


# ---------------------------------------------------------------------------
# GGUF output.
# ---------------------------------------------------------------------------

GGUF_ARCH = "spleeter-unet-v1"


def write_gguf(tensors, output_path):
    import gguf

    writer = gguf.GGUFWriter(output_path, arch=GGUF_ARCH)
    writer.add_string("stemsplit.arch", GGUF_ARCH)
    writer.add_uint32("stemsplit.sample_rate", 44100)
    writer.add_uint32("stemsplit.frame_length", 4096)
    writer.add_uint32("stemsplit.frame_step", 1024)
    writer.add_uint32("stemsplit.n_t", 512)
    writer.add_uint32("stemsplit.n_f", 1024)
    writer.add_uint32("stemsplit.separation_exponent", 2)
    writer.add_array("stemsplit.instruments", list(INSTRUMENTS))

    for name, arr in tensors.items():
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="directory containing the TF checkpoint")
    ap.add_argument("--output", required=True, help="path to write the .gguf file to")
    args = ap.parse_args()

    reader = tf.train.load_checkpoint(args.checkpoint)
    tensors = gather_tensors(reader)

    verify(tensors)

    write_gguf(tensors, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
