"""Execute the released Fullhouse FP32 weights with ONNX Runtime.

No retraining, quantization, feature changes or action-rule changes.
The original encoder and bot.decide remain in use. Only numerical inference
is exported, and probabilities are checked against the NumPy reference.
"""
from pathlib import Path
import numpy as np

def export(model_path, output_path):
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    w = dict(np.load(model_path, allow_pickle=False))
    nodes, initializers = [], []
    def const(name, value):
        initializers.append(nh.from_array(np.asarray(value, dtype=np.float32), name))
        return name
    for name, value in w.items():
        const(name, value.T if name.endswith(('w0', 'w1')) else value)
    const('zero', 0.); const('tiny', 1e-30)
    def op(kind, inputs, output, **kw):
        nodes.append(h.make_node(kind, inputs, [output], **kw))
        return output
    def linear(x, weight, bias, name):
        return op('Add', [op('MatMul', [x, weight], name+'_mm'), bias], name)
    x = linear('features', 'trunk_w0', 'trunk_b0', 'h0')
    x = op('LayerNormalization', [x, 'trunk_ln0_g', 'trunk_ln0_b'], 'ln0', axis=-1, epsilon=1e-5)
    x = op('LeakyRelu', [x], 'act0', alpha=.01)
    x = linear(x, 'trunk_w1', 'trunk_b1', 'h1')
    x = op('LayerNormalization', [x, 'trunk_ln1_g', 'trunk_ln1_b'], 'ln1', axis=-1, epsilon=1e-5)
    x = op('LeakyRelu', [x], 'act1', alpha=.01)
    v = linear(x, 'val_w0', 'val_b0', 'v0')
    v = op('LeakyRelu', [v], 'va', alpha=.01)
    v = linear(v, 'val_w1', 'val_b1', 'value')
    a = linear(x, 'adv_w0', 'adv_b0', 'a0')
    a = op('LeakyRelu', [a], 'aa', alpha=.01)
    a = linear(a, 'adv_w1', 'adv_b1', 'adv')
    a = op('Mul', [a, 'legal'], 'masked_adv')
    count = op('ReduceSum', ['legal'], 'n_legal', keepdims=1)
    mean = op('Div', [op('ReduceSum', [a], 'sum_adv', keepdims=1), count], 'mean_adv')
    mean = op('Mul', [mean, 'legal'], 'masked_mean')
    out = op('Add', [v, a], 'value_adv')
    out = op('Sub', [out, mean], 'centered')
    out = op('Mul', [out, 'legal'], 'masked_output')
    out = op('Relu', [out], 'positive')
    out = op('Mul', [out, 'legal'], 'regrets')
    total = op('ReduceSum', [out], 'total', keepdims=1)
    denom = op('Max', [total, 'tiny'], 'safe_total')
    probs = op('Div', [out, denom], 'normalized')
    fallback = op('Div', ['legal', count], 'uniform_legal')
    positive = op('Greater', [total, 'zero'], 'has_regret')
    op('Where', [positive, probs, fallback], 'probabilities')
    graph = h.make_graph(nodes, 'released_fullhouse_fp32',
                         [h.make_tensor_value_info('features', T.FLOAT, [51]), h.make_tensor_value_info('legal', T.FLOAT, [5])],
                         [h.make_tensor_value_info('probabilities', T.FLOAT, [5])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

class ONNXLookup:
    def __init__(self, model_path):
        import onnxruntime as ort
        from bot.features import load_preflop_equity
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=['CPUExecutionProvider'])
        self.equity = load_preflop_equity()

    def get_strategy(self, state):
        from bot.features import encode_state_dict
        features, legal = encode_state_dict(state, self.equity)
        return self.session.run(None, {'features': features, 'legal': legal.astype(np.float32)})[0]

def install(bot, states):
    target = Path(__file__).with_name('fullhouse_fp32.onnx')
    export(Path(__file__).parent / 'fullhouse-bot/data/deep_cfr_model.npz', target)
    original = bot._DEEP_CFR
    replacement = ONNXLookup(target)
    errors, tvs, agreements = [], [], []
    strict_allclose_failures = 0
    for s in states:
        p, q = original.get_strategy(s), replacement.get_strategy(s)
        assert np.isfinite(q).all() and abs(float(q.sum()) - 1) < 1e-5
        # Record the strict comparison as well: normalized positive regrets
        # can amplify small FP32 accumulation differences near zero.
        strict_allclose_failures += int(not np.allclose(p, q, atol=2e-5, rtol=2e-4))
        errors.append(float(np.max(np.abs(p-q))))
        tvs.append(float(np.abs(p-q).sum()/2))
        agreements.append(bool(p.argmax() == q.argmax()))
    assert max(errors) < 1e-3, f'Probability error too large: {max(errors)}'
    bot._DEEP_CFR = replacement
    return {'model_bytes': target.stat().st_size,
            'parity_states': len(states), 'max_probability_abs_error': max(errors),
            'max_total_variation': max(tvs), 'strict_allclose_failures': strict_allclose_failures,
            'mean_total_variation': float(np.mean(tvs)), 'argmax_agreement': float(np.mean(agreements)),
            'changes': 'FP32 ONNX inference only; released weights, feature encoder, legal mask, sampling and short-stack rules preserved'}
