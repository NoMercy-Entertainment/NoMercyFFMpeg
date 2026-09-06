#!/usr/bin/env python3
"""Compare raw `.f32` layer dumps written by the `stemsplit` filter's
`dump=` option against the reference statistics and sample points in
`fixtures.json` (produced by `dump_reference.py`).

Layout: a dumped `<dir>/<layer>.f32` file is ggml's native `[C][T][F]`
byte order (F fastest-varying), matching `ne = [F, T, C, 1]`. It is read
here as `np.fromfile(...).reshape(C, T, F)`; fixture sample tuples are
`[f, t, c, value]` and are read back from that array as `arr[c, t, f]`.

Usage:
    python compare.py --fixtures fixtures.json --dump <dir> [--rtol 1e-3] \
        [--layers <substring>]

Exit code 0 and "N/N layers match" if every checked layer's shape,
summary statistics and all 64 sample points agree within tolerance; exit
code 1 and one line per failing layer otherwise.
"""

import argparse
import json
import os
import sys

import numpy as np

ATOL = 1e-5  # combined with --rtol for np.isclose; needed near zero, where
# a pure relative tolerance is meaningless (LeakyReLU/sigmoid outputs sit
# at or near 0 often enough that rtol alone would reject correct values).


def check_layer(layer_id, fixture, dump_dir, rtol):
    """Returns None if the layer matches, else a string reason."""
    path = os.path.join(dump_dir, f"{layer_id}.f32")
    if not os.path.exists(path):
        return "missing dump file"

    f, t, c = fixture["shape"]
    data = np.fromfile(path, dtype=np.float32)
    if data.size != f * t * c:
        return f"size mismatch: expected {f * t * c} floats (shape {fixture['shape']}), got {data.size}"
    arr = data.reshape(c, t, f)  # [C][T][F], F fastest -- ggml's native layout

    for stat_name, want in (
        ("mean", fixture["mean"]),
        ("std", fixture["std"]),
        ("min", fixture["min"]),
        ("max", fixture["max"]),
    ):
        got = float(getattr(np, stat_name)(arr))
        if not np.isclose(got, want, rtol=rtol, atol=ATOL):
            return f"{stat_name} mismatch: expected {want!r}, got {got!r}"

    for fi, ti, ci, want in fixture["samples"]:
        got = float(arr[ci, ti, fi])
        if not np.isclose(got, want, rtol=rtol, atol=ATOL):
            return f"sample (f={fi}, t={ti}, c={ci}) mismatch: expected {want!r}, got {got!r}"

    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", required=True, help="path to fixtures.json")
    ap.add_argument("--dump", required=True, help="directory of <layer>.f32 dumps to check")
    ap.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance (default 1e-3)")
    ap.add_argument(
        "--layers",
        default=None,
        help="only check layer ids containing this substring (e.g. 'conv')",
    )
    args = ap.parse_args()

    with open(args.fixtures) as fh:
        fixtures = json.load(fh)

    layer_ids = [lid for lid in fixtures if args.layers is None or args.layers in lid]
    if not layer_ids:
        print(f"no fixture layers match --layers {args.layers!r}", file=sys.stderr)
        return 1

    failures = []
    for layer_id in layer_ids:
        reason = check_layer(layer_id, fixtures[layer_id], args.dump, args.rtol)
        if reason is not None:
            failures.append((layer_id, reason))
            print(f"FAIL {layer_id}: {reason}")

    n_total = len(layer_ids)
    n_pass = n_total - len(failures)
    print(f"{n_pass}/{n_total} layers match")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
