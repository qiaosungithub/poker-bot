"""Compare JAX forward, losses, gradients and Adam to the PyTorch reference."""
import launch
import json, time
import numpy as np
import torch
import jax_backend as jb
from training.network import DuelingAdvantageNet, AverageStrategyNet


def main():
    torch.set_num_threads(1)
    torch.manual_seed(77)
    rng = np.random.default_rng(77)
    x = rng.normal(size=(128, 51)).astype(np.float32)
    x[:, 46:] = rng.integers(0, 2, size=(128, 5))
    x[:, 47] = 1
    y = rng.normal(size=(128, 5)).astype(np.float32)
    w = rng.uniform(size=128).astype(np.float32)
    records = []
    for strategy in (False, True):
        net = AverageStrategyNet() if strategy else DuelingAdvantageNet()
        if strategy:
            y = np.abs(y)
        p = jb.params_from_torch(net)
        xt, yt, wt = map(torch.from_numpy, (x, y, w))
        mask = xt[:, 46:]
        expected = net(xt, mask)
        actual = np.asarray(jb.forward(p, jb.jnp.asarray(x), jb.jnp.asarray(x[:, 46:]), strategy))
        np.testing.assert_allclose(actual, expected.detach().numpy(), atol=2e-5, rtol=2e-4)
        if strategy:
            target = yt * mask
            target = target / target.sum(1, keepdim=True).clamp(min=1e-12)
            loss = (-(target * torch.log_softmax(expected, -1)).sum(-1) * wt).mean()
        else:
            n = mask.sum(1, keepdim=True).clamp(min=1)
            mean = (yt * mask).sum(1, keepdim=True) / n
            var = (((yt * mask - mean * mask)**2)*mask).sum(1, keepdim=True) / n
            target = (yt-mean) / torch.sqrt(var+1e-6) * mask
            loss = (((expected-target)**2).mean(-1)*wt).mean()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss.backward()
        jl, grad = jb.jax.value_and_grad(jb.objective)(p, jb.jnp.asarray(x), jb.jnp.asarray(y), jb.jnp.asarray(w), strategy)
        np.testing.assert_allclose(float(jl), float(loss.detach()), atol=2e-5)
        grad_error = max(float(np.max(np.abs(np.asarray(grad[k])-v.grad.numpy()))) for k,v in net.named_parameters())
        assert grad_error < 2e-5, grad_error
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()
        tx = jb.optax.scale_by_adam()
        updates, _ = tx.update(jb.clip_torch(grad), tx.init(p), p)
        jp = jb.jax.tree.map(lambda a,u:a-.001*u, p, updates)
        update_error = max(float(np.max(np.abs(np.asarray(jp[k])-v.detach().numpy()))) for k,v in net.named_parameters())
        assert update_error < 2e-5, update_error
        # Negative control: a changed policy-head weight must fail parity.
        corrupt = dict(p); name = 'policy_head.weight' if strategy else 'advantage_head.2.weight'
        corrupt[name] = corrupt[name].at[0, 0].add(10.)
        wrong = np.asarray(jb.forward(corrupt, jb.jnp.asarray(x), jb.jnp.asarray(x[:,46:]), strategy))
        assert not np.allclose(wrong, actual, atol=2e-5)
        records.append(dict(strategy=strategy, forward_max_error=float(np.max(np.abs(actual-expected.detach().numpy()))),
            loss=float(jl), gradient_max_error=grad_error, adam_max_error=update_error))
    result = dict(devices=[str(d) for d in jb.jax.devices()], records=records)
    (launch.ROOT/'backend_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
