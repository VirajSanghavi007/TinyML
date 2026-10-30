"""Export a QAT TinyNet to a fully-int8 .tflite, shrink it, and verify it with the TFLite interpreter.

  uv run --group export python -m tinyml.export_tflite --run-dir runs/v2 --out firmware/model

Model contract (what the firmware sees):
  input  int8 [1,32,32,3] NHWC = raw RGB pixel - 128 (CIFAR normalization is folded into the stem conv)
  output int8 [1,10] softmax probabilities

The network goes torch -> a small IR (numpy weights, BN folded) -> Keras -> TFLiteConverter (int8).
Padding: models trained with pad mode "tf" use plain 'same' convs, so the graph has no PAD op and the
stem's border matches torch exactly (see ofa.PAD_MODES). Legacy "torch" models get an explicit
ZeroPadding2D + 'valid' on stride-2 convs, which the converter turns into PAD ops.

Patch-based inference (MCUNetV2 style, `patch=(n_units, n_stripes)`): the first n_units units (stem,
then MBConv blocks) run once per horizontal stripe of their output, each stripe cropping just the
input rows it needs (receptive-field halo included) and zero-padding only at real image borders. The
stripes are concatenated, then the rest of the network runs on the full tensor. Peak activation memory
drops because the high-resolution tensors only ever exist one stripe at a time.

After conversion: tensor names are stripped (strip_names) and selected conv/FC layers can be rewritten
from per-channel to per-tensor weight quantization (to_per_tensor), which shrinks the flatbuffer's
quantization tables. Neither changes the TFLM arena."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .ofa import TinyNet
from .qat import QTinyNet
from .tfl_build import (PIXEL_OFFSET, _conv_ir, build_keras, convert, flatbuffer_breakdown,  # noqa: F401
                        ir_from_spec, make_interpreter, model_report, prefix_rows, same_pad, stripe_bounds,
                        strip_names, to_nhwc_offset, to_per_tensor, units, weight_ops)


# ---------------------------------------------------------------- torch -> IR
def load_qat_model(run_dir, spec_file="pruned_ft.pt", qat_file="qat_int8_fakequant.pt"):
    run_dir = Path(run_dir)
    spec = torch.load(run_dir / spec_file, map_location="cpu", weights_only=False)["spec"]
    q = QTinyNet(TinyNet(spec))
    q.load_state_dict(torch.load(run_dir / qat_file, map_location="cpu"))
    return q.eval()


def fold_input_normalization(w, b):
    """Rewrite stem conv (OIHW) so it takes (pixel - 128) instead of (pixel/255 - mean)/std.

    x_norm = (p' + 128)/255/std - mean/std = p' * a + c   with a = 1/(255 std), c = (128/255 - mean)/std
    Exact inside the image. At the border it is exact only for pad mode "tf", whose stem pads with
    the normalized value of pixel 128 (= int8 0); "torch" mode pads the normalized input with 0."""
    mean, std = np.array(D.CIFAR_MEAN), np.array(D.CIFAR_STD)
    a = 1.0 / (255.0 * std)
    c = (PIXEL_OFFSET / 255.0 - mean) / std
    w2 = w * a[None, :, None, None]
    b2 = b + (w * c[None, :, None, None]).sum(axis=(1, 2, 3))
    return w2.astype(np.float32), b2.astype(np.float32)


def ir_from_qat(q):
    """IR of a QTinyNet: BN already folded by QConv; the stem also absorbs the input normalization."""
    npy = lambda t: t.detach().cpu().numpy()  # noqa: E731

    def conv(qc, w=None, b=None):
        return _conv_ir(npy(qc.weight) if w is None else w, npy(qc.bias) if b is None else b,
                        qc.k, qc.stride[0], qc.groups, qc.relu6)
    sw, sb = fold_input_normalization(npy(q.stem.weight), npy(q.stem.bias))
    return {"pad": q.stem.pad_mode, "stem": conv(q.stem, sw, sb),
            "blocks": [{"expand": conv(b.expand), "dw": conv(b.dw), "proj": conv(b.proj), "res": b.use_res}
                       for b in q.blocks],
            "head": conv(q.head), "fc_w": npy(q.fc_w), "fc_b": npy(q.fc_b)}


def quantize_input(x_nhwc_offset, scale, zp):
    return np.clip(np.round(x_nhwc_offset / scale + zp), -128, 127).astype(np.int8)


def tflite_predict(it, x_u8):
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, zp = inp["quantization"]
    xs = to_nhwc_offset(x_u8)
    preds = np.empty(len(xs), np.int64)
    for i in range(len(xs)):
        it.set_tensor(inp["index"], quantize_input(xs[i:i + 1], s, zp))
        it.invoke()
        preds[i] = it.get_tensor(out["index"])[0].argmax()
    return preds


def tflite_accuracy(it, x_u8, y, log=print, chunks=10):
    correct, n, t0 = 0, len(x_u8), time.time()
    edges = np.linspace(0, n, chunks + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        correct += int((tflite_predict(it, x_u8[a:b]) == y[a:b]).sum())
        el = time.time() - t0
        log(f"[tflite eval] {b}/{n} ({100 * b / n:.0f}%) acc={correct / max(b, 1):.4f} "
            f"- elapsed {el:.0f}s, eta {el / max(b, 1) * (n - b):.0f}s")
    return correct / n


def choose_per_tensor(tflite_bytes, x_val_u8, y_val, max_drop=0.002, log=print):
    """Greedy: switch conv/FC layers to per-tensor weights, largest flash saving first, keeping each
    switch only if validation accuracy stays within `max_drop` of the all-per-channel model.
    Returns (model_bytes, chosen_op_indices, info)."""
    base = tflite_accuracy(make_interpreter(tflite_bytes), x_val_u8, y_val, log=lambda m: None)
    cands = sorted(weight_ops(tflite_bytes), key=lambda t: -t[2])
    chosen, cur, acc = [], tflite_bytes, base
    for n, (i, name, ch) in enumerate(cands):
        trial = to_per_tensor(tflite_bytes, chosen + [i])
        a = tflite_accuracy(make_interpreter(trial), x_val_u8, y_val, log=lambda m: None)
        keep = a >= base - max_drop
        if keep:
            chosen, cur, acc = chosen + [i], trial, a
        log(f"[per-tensor] {n + 1}/{len(cands)} ({100 * (n + 1) / len(cands):.0f}%) op {i} {name} ch={ch}: "
            f"val {a:.4f} (base {base:.4f}) -> {'keep' if keep else 'revert'}; model {len(cur)} B")
    return cur, chosen, {"val_base": base, "val_final": acc, "per_tensor_ops": chosen,
                         "candidates": len(cands)}


# ---------------------------------------------------------------- outputs
def write_tflite(tflite_bytes, out_dir):
    """Only the .tflite: firmware/make_model_cc.sh turns it into model.cc with `xxd -i`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.tflite").write_bytes(tflite_bytes)


def write_test_images(it, x_u8, y, out_dir, n):
    """Emit n test images (int8 NHWC, input-quantized) + labels + the host interpreter's int8 outputs,
    so the firmware can check its own outputs against the host bit for bit."""
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    s, zp = inp["quantization"]
    xs = quantize_input(to_nhwc_offset(x_u8[:n]), s, zp)
    outs = []
    for i in range(n):
        it.set_tensor(inp["index"], xs[i:i + 1])
        it.invoke()
        outs.append(it.get_tensor(out["index"])[0].copy())
    arr = lambda a: ", ".join(str(int(v)) for v in np.asarray(a).ravel())  # noqa: E731
    body = [f"#include \"test_images.h\"\n\nconst int kNumTestImages = {n};\n",
            f"alignas(16) const int8_t kTestImages[{n}][{xs[0].size}] = {{"]
    body += [f"  {{{arr(img)}}}," for img in xs]
    body += ["};", f"const uint8_t kTestLabels[{n}] = {{{arr(y[:n])}}};",
             f"const int8_t kHostOutputs[{n}][{outs[0].size}] = {{"]
    body += [f"  {{{arr(o)}}}," for o in outs]
    body += ["};", ""]
    Path(out_dir, "test_images.cc").write_text("\n".join(body))
    Path(out_dir, "test_images.h").write_text(
        "#ifndef TEST_IMAGES_H_\n#define TEST_IMAGES_H_\n#include <stdint.h>\n\n"
        f"extern const int kNumTestImages;\nextern const int8_t kTestImages[][{xs[0].size}];\n"
        f"extern const uint8_t kTestLabels[];\nextern const int8_t kHostOutputs[][{outs[0].size}];\n\n"
        "#endif  // TEST_IMAGES_H_\n")


def write_op_header(ops, out_dir):
    from .tflm_cost import resolver_header
    Path(out_dir, "model_ops.h").write_text(resolver_header(ops))


def port_check(q, ir, x_u8, patch=None):
    """Float torch (fake-quant off) vs float Keras logits on a few images: validates the port."""
    with torch.no_grad():
        for m in q.modules():
            if hasattr(m, "disable_fake_quant"):
                m.disable_fake_quant()
        xn = D.GPULoader(x_u8, np.zeros(len(x_u8), np.int64), len(x_u8), "cpu").normalize(torch.as_tensor(x_u8))
        torch_logits = q(xn).numpy()
        for m in q.modules():
            if hasattr(m, "enable_fake_quant"):
                m.enable_fake_quant()
    km = build_keras(ir, softmax=False, patch=patch)
    xs = to_nhwc_offset(x_u8)
    keras_logits = np.concatenate([km(xs[i:i + 1], training=False).numpy() for i in range(len(xs))])
    return {"max_abs_logit_diff": float(np.abs(torch_logits - keras_logits).max()),
            "argmax_agreement": float((torch_logits.argmax(1) == keras_logits.argmax(1)).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/main")
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="firmware/model")
    ap.add_argument("--n-test", type=int, default=0, help="0 = full 10k test set")
    ap.add_argument("--n-val-per-tensor", type=int, default=2000, help="val images for the per-tensor greedy")
    ap.add_argument("--n-firmware-images", type=int, default=8)
    ap.add_argument("--patch", default="auto", help="'auto' (from deploy.json), 'none', or N_UNITS,N_STRIPES")
    ap.add_argument("--per-tensor", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)

    q = load_qat_model(run_dir)
    patch = None
    if args.patch == "auto" and (run_dir / "deploy.json").exists():
        patch = json.loads((run_dir / "deploy.json").read_text()).get("patch")
    elif args.patch not in ("auto", "none"):
        patch = [int(v) for v in args.patch.split(",")]
    patch = tuple(patch) if patch else None
    xtr, ytr, xte, yte = D.load_cifar10(args.data)
    xtr, ytr, xva, yva = D.split_train_val(xtr, ytr, n_val=5000, seed=0)  # same split as training
    if args.n_test:
        xte, yte = xte[:args.n_test], yte[:args.n_test]

    ir = ir_from_qat(q)
    port = port_check(q, ir, xte[:256], patch)
    print("torch vs keras float port:", port, flush=True)
    full = convert(build_keras(ir, patch=patch), xtr)
    stripped = strip_names(full)
    sizes = {"converted": len(full), "names_stripped": len(stripped)}
    tfl, pt_info = stripped, None
    if args.per_tensor:
        tfl, _, pt_info = choose_per_tensor(stripped, xva[:args.n_val_per_tensor], yva[:args.n_val_per_tensor])
        sizes["per_tensor"] = len(tfl)
    it = make_interpreter(tfl)
    rep = model_report(it)
    rep.update({"patch": list(patch) if patch else None, "sizes": sizes, "per_tensor": pt_info,
                "flash_breakdown": flatbuffer_breakdown(tfl)})
    print(json.dumps(rep, indent=2), flush=True)
    acc = tflite_accuracy(it, xte, yte)
    rep.update({"tflite_int8_test_acc": acc, "n_test": len(xte), "model_bytes": len(tfl), "port_check": port})
    write_tflite(tfl, args.out)
    write_test_images(it, xte, yte, args.out, args.n_firmware_images)
    write_op_header(rep["ops"], args.out)
    Path(args.out, "export_report.json").write_text(json.dumps(rep, indent=2))
    print(f"int8 TFLite test accuracy: {acc:.4f} on {len(xte)} images; model {len(tfl)} bytes -> {args.out}")


if __name__ == "__main__":
    main()
