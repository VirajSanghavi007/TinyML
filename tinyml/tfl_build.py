"""Torch-free half of the export: IR -> Keras -> int8 TFLite, plus flatbuffer edits and the
patch-based (striped) graph builder. Imports TensorFlow lazily, so pool workers that only build and
convert candidate models (tinyml/tflm_cost.py) never load torch.

IR = {"pad": "tf"|"torch", "stem": conv, "blocks": [{"expand", "dw", "proj": conv, "res": bool}],
      "head": conv, "fc_w": [out, in], "fc_b": [out]}, conv = {"w" OIHW, "b", "k", "s", "groups", "relu6"}."""
import numpy as np

PIXEL_OFFSET = 128.0


def same_pad(n, k, s):
    """TF 'same' padding (before, after) for one spatial dim of size n."""
    total = max((-(-n // s) - 1) * s + k - n, 0)
    return total // 2, total - total // 2


def _conv_ir(w, b, k, s, groups, relu6):
    return {"w": np.asarray(w, np.float32), "b": np.asarray(b, np.float32), "k": int(k), "s": int(s),
            "groups": int(groups), "relu6": bool(relu6)}


def ir_from_spec(spec, seed=0):
    """IR with random weights for a spec: for measuring deployment cost, which depends only on shapes."""
    rng = np.random.default_rng(seed)

    def conv(cin, cout, k, s, groups, relu6):
        fan_in = (cin // groups) * k * k
        w = rng.normal(0, np.sqrt(2.0 / fan_in), (cout, cin // groups, k, k))
        return _conv_ir(w, rng.normal(0, 0.1, cout), k, s, groups, relu6)
    blocks = [{"expand": conv(b["cin"], b["hidden"], 1, 1, 1, True),
               "dw": conv(b["hidden"], b["hidden"], b["k"], b["stride"], b["hidden"], True),
               "proj": conv(b["hidden"], b["cout"], 1, 1, 1, False),
               "res": b["stride"] == 1 and b["cin"] == b["cout"]} for b in spec["blocks"]]
    cin = spec["blocks"][-1]["cout"] if spec["blocks"] else spec["stem"]
    return {"pad": spec.get("pad", "torch"), "stem": conv(3, spec["stem"], 3, 2, 1, True), "blocks": blocks,
            "head": conv(cin, spec["head"], 1, 1, 1, True),
            "fc_w": rng.normal(0, np.sqrt(1.0 / spec["head"]), (spec["num_classes"], spec["head"])),
            "fc_b": np.zeros(spec["num_classes"])}


def units(ir):
    """Prefix units in execution order: the stem, then each MBConv block."""
    return [("stem", ir["stem"])] + [("block", b) for b in ir["blocks"]]


def unit_convs(unit):
    kind, u = unit
    return [u] if kind == "stem" else [u["expand"], u["dw"], u["proj"]]


def conv_in_rows(o0, o1, n_in, k, s):
    """Input rows [i0, i1) (may extend past [0, n_in) into padding) needed for output rows [o0, o1)
    of a TF 'same' conv."""
    before, _ = same_pad(n_in, k, s)
    return o0 * s - before, (o1 - 1) * s - before + k


def stripe_bounds(n, n_stripes):
    edges = [round(i * n / n_stripes) for i in range(n_stripes + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n_stripes) if edges[i + 1] > edges[i]]


def prefix_rows(ir, n_units, o0, o1, res=32):
    """Back-propagate an output row range of the first n_units through them.

    Returns (input_rows, plan): plan[u] = list of per-conv (in_rows, out_rows, n_in) for unit u, plus the
    unit's input/output rows, so the builder knows what to crop and pad. Rows beyond the tensor are padding."""
    sizes = [res]
    for unit in units(ir)[:n_units]:
        n = sizes[-1]
        for c in unit_convs(unit):
            n = -(-n // c["s"])
        sizes.append(n)
    plan = [None] * n_units
    rows = (o0, o1)
    for ui in range(n_units - 1, -1, -1):
        unit = units(ir)[ui]
        convs = unit_convs(unit)
        n = sizes[ui]
        ns = [n]
        for c in convs:
            ns.append(-(-ns[-1] // c["s"]))
        out_rows = rows
        per_conv = []
        r = out_rows
        for c, n_in in zip(reversed(convs), reversed(ns[:-1])):
            i0, i1 = conv_in_rows(r[0], r[1], n_in, c["k"], c["s"])
            per_conv.append(((i0, i1), r, n_in))
            r = (max(i0, 0), min(i1, n_in))  # rows that must exist in the producer's tensor
        per_conv.reverse()
        in_rows = r
        if unit[0] == "block" and unit[1]["res"]:
            in_rows = (min(in_rows[0], out_rows[0]), max(in_rows[1], out_rows[1]))
        plan[ui] = {"in_rows": in_rows, "out_rows": out_rows, "convs": per_conv}
        rows = in_rows
    return rows, plan, sizes


# ---------------------------------------------------------------- IR -> Keras
def build_keras(ir, softmax=True, patch=None, res=32):
    """Keras model for the IR. patch=(n_units, n_stripes) runs the first n_units per stripe (pad "tf" only)."""
    import tensorflow as tf
    L = tf.keras.layers
    tf_pad = ir["pad"] == "tf"
    made = {}

    def layer_for(c, padding):
        key = (id(c), padding)
        if key not in made:
            act = "relu6" if c["relu6"] else None
            if c["groups"] > 1:
                made[key] = L.DepthwiseConv2D(c["k"], c["s"], padding, activation=act)
            else:
                made[key] = L.Conv2D(c["w"].shape[0], c["k"], c["s"], padding, activation=act)
        return made[key]

    def set_weights():
        for (cid, _), layer in made.items():
            c = by_id[cid]
            if c["groups"] > 1:
                layer.set_weights([c["w"].transpose(2, 3, 0, 1), c["b"]])  # [C,1,k,k] -> [k,k,C,1]
            else:
                layer.set_weights([c["w"].transpose(2, 3, 1, 0), c["b"]])  # OIHW -> HWIO
    by_id = {id(c): c for unit in units(ir) for c in unit_convs(unit)} | {id(ir["head"]): ir["head"]}

    def conv_full(x, c):
        if tf_pad or c["s"] == 1:
            return layer_for(c, "same")(x)
        x = L.ZeroPadding2D(c["k"] // 2)(x)  # torch-mode stride 2: symmetric pad, becomes a PAD op
        return layer_for(c, "valid")(x)

    def unit_full(x, unit):
        if unit[0] == "stem":
            return conv_full(x, unit[1])
        b = unit[1]
        y = conv_full(conv_full(conv_full(x, b["expand"]), b["dw"]), b["proj"])
        return L.Add()([x, y]) if b["res"] else y

    def crop_rows(x, have, want):
        top, bottom = want[0] - have[0], have[1] - want[1]
        assert top >= 0 and bottom >= 0, (have, want)
        return L.Cropping2D(((top, bottom), (0, 0)))(x) if top or bottom else x

    def conv_patch(x, have, c, step):
        (i0, i1), _, n_in = step
        need = (max(i0, 0), min(i1, n_in))
        x = crop_rows(x, have, need)
        t, b = max(0, -i0), max(0, i1 - n_in)
        l, r = same_pad(x.shape[2], c["k"], c["s"])  # columns are never split: plain 'same' padding
        if t or b or l or r:
            x = L.ZeroPadding2D(((t, b), (l, r)))(x)
        return layer_for(c, "valid")(x)

    inp = tf.keras.Input((res, res, 3), batch_size=1, name="pixels_minus_128")
    us = units(ir)
    x, start = inp, 0
    if patch:
        n_units, n_stripes = patch
        if not tf_pad:
            raise ValueError("patch-based inference needs a model trained with pad mode 'tf'")
        _, _, sizes = prefix_rows(ir, n_units, 0, 1, res)
        parts = []
        for o0, o1 in stripe_bounds(sizes[n_units], n_stripes):
            in_rows, plan, _ = prefix_rows(ir, n_units, o0, o1, res)
            y, have = crop_rows(inp, (0, res), in_rows), in_rows
            for unit, p in zip(us[:n_units], plan):
                y = crop_rows(y, have, p["in_rows"])
                have = p["in_rows"]
                u_in = y
                for c, step in zip(unit_convs(unit), p["convs"]):
                    y = conv_patch(y, have, c, step)
                    have = step[1]
                if unit[0] == "block" and unit[1]["res"]:
                    y = L.Add()([crop_rows(u_in, p["in_rows"], p["out_rows"]), y])
                have = p["out_rows"]
            parts.append(y)
        x = L.Concatenate(axis=1)(parts) if len(parts) > 1 else parts[0]
        start = n_units
    for unit in us[start:]:
        x = unit_full(x, unit)
    x = conv_full(x, ir["head"])
    x = L.GlobalAveragePooling2D()(x)
    fc = L.Dense(ir["fc_w"].shape[0])
    x = fc(x)
    if softmax:
        x = L.Softmax()(x)
    model = tf.keras.Model(inp, x)
    set_weights()
    fc.set_weights([np.asarray(ir["fc_w"], np.float32).T, np.asarray(ir["fc_b"], np.float32)])
    return model


def to_nhwc_offset(x_u8_nchw):
    return (x_u8_nchw.transpose(0, 2, 3, 1).astype(np.float32) - PIXEL_OFFSET)


def convert(keras_model, calib_u8, n_calib=1000):
    import tensorflow as tf
    calib = to_nhwc_offset(calib_u8[:n_calib])

    def rep():
        for i in range(len(calib)):
            yield [calib[i:i + 1]]
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def strip_names(tflite_bytes):
    """Drop tensor/signature name strings (only debuggers and Netron read them; TFLM uses indices)."""
    from tensorflow.lite.tools import flatbuffer_utils as fu
    m = fu.convert_bytearray_to_object(bytearray(tflite_bytes))
    fu.strip_strings(m)
    return bytes(fu.convert_object_to_bytearray(m))


WEIGHT_OPS = {"CONV_2D": 0, "DEPTHWISE_CONV_2D": 3, "FULLY_CONNECTED": 0}  # op -> filter output-channel axis


def _opname(m, op):
    from tensorflow.lite.python import schema_py_generated as S
    names = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
    oc = m.operatorCodes[op.opcodeIndex]
    return names[max(oc.builtinCode, oc.deprecatedBuiltinCode)]


def weight_ops(tflite_bytes):
    """[(op_index, op_name, n_channels)] for every conv/depthwise/FC op with per-channel weights."""
    from tensorflow.lite.tools import flatbuffer_utils as fu
    m = fu.convert_bytearray_to_object(bytearray(tflite_bytes))
    g, out = m.subgraphs[0], []
    for i, op in enumerate(g.operators):
        name = _opname(m, op)
        if name in WEIGHT_OPS:
            q = g.tensors[op.inputs[1]].quantization
            n = 0 if q is None or q.scale is None else len(q.scale)
            if n > 1:
                out.append((i, name, n))
    return out


def to_per_tensor(tflite_bytes, op_indices):
    """Requantize the filters of `op_indices` from per-channel to one per-tensor scale (and their int32
    biases to match). Saves 12 B of scale/zero-point per channel of filter and of bias in flash."""
    from tensorflow.lite.tools import flatbuffer_utils as fu
    m = fu.convert_bytearray_to_object(bytearray(tflite_bytes))
    g = m.subgraphs[0]
    for i in op_indices:
        op = g.operators[i]
        axis = WEIGHT_OPS[_opname(m, op)]
        in_t, w_t = g.tensors[op.inputs[0]], g.tensors[op.inputs[1]]
        wq = w_t.quantization
        scales = np.asarray(wq.scale, np.float64)
        if len(scales) <= 1:
            continue
        wbuf = m.buffers[w_t.buffer]
        w = np.frombuffer(bytes(wbuf.data), np.int8).reshape(w_t.shape).astype(np.float64)
        shape = [1] * w.ndim
        shape[axis] = -1
        real = w * scales.reshape(shape)
        s = float(np.abs(real).max() / 127.0) or 1.0
        wbuf.data = np.clip(np.round(real / s), -127, 127).astype(np.int8).view(np.uint8)
        wq.scale, wq.zeroPoint, wq.quantizedDimension = [s], [0], 0
        if len(op.inputs) > 2 and op.inputs[2] >= 0:
            b_t = g.tensors[op.inputs[2]]
            bbuf = m.buffers[b_t.buffer]
            in_s = float(in_t.quantization.scale[0])
            b = np.frombuffer(bytes(bbuf.data), np.int32).astype(np.float64) * in_s * scales
            bbuf.data = np.round(b / (in_s * s)).astype(np.int32).view(np.uint8)
            b_t.quantization.scale, b_t.quantization.zeroPoint = [in_s * s], [0]
            b_t.quantization.quantizedDimension = 0
    return bytes(fu.convert_object_to_bytearray(m))


# ---------------------------------------------------------------- interpreter checks
def make_interpreter(tflite_bytes):
    import tensorflow as tf
    # No default delegates: XNNPACK would rewrite the graph and show up as a DELEGATE op,
    # hiding the real builtin ops the firmware's op resolver has to register.
    it = tf.lite.Interpreter(model_content=tflite_bytes, experimental_op_resolver_type=
                             tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    return it


def model_report(it):
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    ops = sorted({d["op_name"] for d in it._get_ops_details()})
    float_tensors = [t["name"] for t in it.get_tensor_details() if t["dtype"] == np.float32]
    return {
        "input": {"dtype": np.dtype(inp["dtype"]).name, "shape": inp["shape"].tolist(),
                  "scale": float(inp["quantization"][0]), "zero_point": int(inp["quantization"][1])},
        "output": {"dtype": np.dtype(out["dtype"]).name, "shape": out["shape"].tolist(),
                   "scale": float(out["quantization"][0]), "zero_point": int(out["quantization"][1])},
        "ops": ops,
        "op_count": len(it._get_ops_details()),
        "float32_tensors": float_tensors,
    }


def flatbuffer_breakdown(tflite_bytes):
    """Where the .tflite bytes go: weight/bias payload vs quantization params vs everything else."""
    from tensorflow.lite.python import schema_py_generated as S
    m = S.Model.GetRootAsModel(tflite_bytes, 0)
    payload = {"int8": 0, "int32": 0, "other": 0}
    qparams = names = 0
    g = m.Subgraphs(0)
    for i in range(g.TensorsLength()):
        t = g.Tensors(i)
        names += len(t.Name() or b"")
        q = t.Quantization()
        if q is not None:
            qparams += q.ScaleLength() * 4 + q.ZeroPointLength() * 8  # float32 scale, int64 zero point
        n = m.Buffers(t.Buffer()).DataLength()
        kind = {S.TensorType.INT8: "int8", S.TensorType.INT32: "int32"}.get(t.Type(), "other")
        payload[kind] += n
    total = len(tflite_bytes)
    used = sum(payload.values()) + qparams + names
    return {"total": total, "int8_weights": payload["int8"], "int32_biases": payload["int32"],
            "other_const": payload["other"], "quant_params": qparams, "tensor_names": names,
            "flatbuffer_structure_and_metadata": total - used}


