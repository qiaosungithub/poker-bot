"""TEST ORACLE ONLY: fp32 JAX reference of the upstream network-fitting objective.

This is the fitting code of the original reproduction (``fullhouse-repro/jax_backend.py``
on ``main``, commit 7356b42), kept verbatim apart from this docstring and a log label so
the GPU fit can be checked against it (``test_gpu_fit.py`` T2/T5, ``validate_fit.py``).
It runs on CPU or GPU JAX; nothing in the GPU trainer imports it.
"""
import functools
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update('jax_default_matmul_precision', 'highest')


def params_from_torch(net):
    return {k: jnp.asarray(v.detach().cpu().numpy()) for k, v in net.state_dict().items()}


def forward(p, x, mask, strategy=False):
    def linear(x, prefix):
        return x @ p[prefix + '.weight'].T + p[prefix + '.bias']
    for i in (0, 3):
        x = linear(x, f'trunk.{i}')
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + 1e-5)
        x = x * p[f'trunk.{i+1}.weight'] + p[f'trunk.{i+1}.bias']
        x = jax.nn.leaky_relu(x, 0.01)
    if strategy:
        return jnp.where(mask > 0, linear(x, 'policy_head'), -1e9)
    v = linear(jax.nn.leaky_relu(linear(x, 'value_head.0'), .01), 'value_head.2')
    a = linear(jax.nn.leaky_relu(linear(x, 'advantage_head.0'), .01), 'advantage_head.2') * mask
    mean = a.sum(-1, keepdims=True) / jnp.maximum(mask.sum(-1, keepdims=True), 1)
    return (v + a - mean * mask) * mask


def objective(p, x, y, w, strategy=False):
    mask = x[:, 46:51]
    pred = forward(p, x, mask, strategy)
    if strategy:
        y = y * mask
        y = y / jnp.maximum(y.sum(-1, keepdims=True), 1e-12)
        loss = -(y * jax.nn.log_softmax(pred)).sum(-1)
    else:
        n = jnp.maximum(mask.sum(-1, keepdims=True), 1)
        mean = (y * mask).sum(-1, keepdims=True) / n
        var = (((y * mask - mean * mask) ** 2) * mask).sum(-1, keepdims=True) / n
        y = (y - mean) / jnp.sqrt(var + 1e-6) * mask
        loss = ((pred - y) ** 2).mean(-1)
    return (loss * w).mean()


def clip_torch(grads):
    norm = jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1., 1. / (norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads)


@functools.partial(jax.jit, static_argnames=('strategy', 'batch_size', 'steps'))
def fit_chunk(p, opt_state, key, data, sizes, lr, *, strategy, batch_size, steps):
    tx = optax.scale_by_adam(b1=.9, b2=.999, eps=1e-8)
    offsets = jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.cumsum(sizes)[:-1]))
    probabilities = sizes / sizes.sum()
    def step(carry, _):
        p, opt_state, key = carry
        key, kb, ki = jax.random.split(key, 3)
        # Match CPU ReservoirBuffer sampling: choose one player per minibatch,
        # proportional to buffer size, then sample uniformly within that player.
        b = jax.random.categorical(kb, jnp.log(probabilities))
        ix = offsets[b] + jax.random.randint(ki, (batch_size,), 0, sizes[b])
        rows = data[ix]
        x, y, iteration, reach = rows[:, :51], rows[:, 51:56], rows[:, 56], rows[:, 57]
        w = (iteration / jnp.maximum(iteration.max(), 1.)) ** 1.5
        if strategy:
            w = w * reach
        loss, grads = jax.value_and_grad(objective)(p, x, y, w, strategy)
        updates, opt_state = tx.update(clip_torch(grads), opt_state, p)
        p = jax.tree.map(lambda v, u: v - lr * u, p, updates)
        return (p, opt_state, key), loss
    (p, opt_state, key), losses = jax.lax.scan(step, (p, opt_state, key), None, length=steps)
    return p, opt_state, key, losses.mean()


def fit(net, buffers, optimizer, n_steps, batch_size, strategy):
    import torch
    start = time.monotonic()
    sizes = np.array([len(b) for b in buffers], np.int32)
    # Never draw a larger batch than the smallest occupied buffer: upstream
    # sample() caps batch size. Formal configurations have ample samples.
    batch_size = min(batch_size, int(sizes[sizes > 0].min()))
    total = int(sizes.sum())
    capacity = 1 << (total - 1).bit_length()
    host = np.zeros((capacity, 58), np.float32)
    offset = 0
    for b in buffers:
        n = len(b)
        host[offset:offset+n, :51] = b.features[:n]
        host[offset:offset+n, 51:56] = b.advantages[:n]
        host[offset:offset+n, 56] = b.iterations[:n]
        host[offset:offset+n, 57] = b.sample_weights[:n]
        offset += n
    data = jax.device_put(host)
    del host
    p = params_from_torch(net)
    state = optax.scale_by_adam().init(p)
    key = jax.random.PRNGKey(int(np.random.randint(0, 2**31-1)))
    lr = jnp.float32(optimizer.param_groups[0]['lr'])
    loss_sum = 0.
    for start_step in range(0, n_steps, 250):
        count = min(250, n_steps - start_step)
        p, state, key, loss = fit_chunk(p, state, key, data, jnp.asarray(sizes), lr,
            strategy=strategy, batch_size=batch_size, steps=count)
        loss_sum += float(loss) * count
        if start_step == 0 or (start_step + count) % 2500 == 0:
            print(f'[JAX ref fit] strategy={strategy} step={start_step+count}/{n_steps} '
                  f'loss={float(loss):.5f} seconds={time.monotonic()-start:.1f}', flush=True)
    net.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in p.items()})
    net.eval()
    return loss_sum / n_steps


def train_network(net, buffers, optimizer, n_steps=4000, batch_size=32768,
                  device=None, gpu_buffer=None):
    return fit(net, buffers, optimizer, n_steps, batch_size, False)


def train_strategy_network(net, strategy_buffer, optimizer, n_steps=3000,
                           batch_size=8192, device=None, milestones=None, tag=''):
    return fit(net, [strategy_buffer], optimizer, n_steps, batch_size, True)
