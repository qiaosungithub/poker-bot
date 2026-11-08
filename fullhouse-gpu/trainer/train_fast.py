"""Fast, faithful Fullhouse Deep CFR trainer (upstream ``training/train.py::train_deep_cfr``).

Same algorithm and hyper-parameters as upstream (+ ``training_changes.patch``) as
launched by the original TPU reproduction (``fullhouse-repro/launch.py
config_main.json`` on ``main``, removed on this branch); only *where* the work runs
changes:

* traversal: ``../traversal``'s level-synchronous batched traversal, fp32 'highest'
  GPU forward on ``devices.trav`` (numba engine on all CPU cores);
* reservoirs: ``../fit``'s GPU-resident ``GPUReservoirSet`` (6 x 5M advantage +
  5M strategy rows) on ``devices.fit`` -- no per-iteration host rebuild/upload;
* advantage fit: ``../fit``'s CUDA-graph fitter on ``devices.fit``;
* strategy fit: only at checkpoint iterations, on ``devices.strat``, overlapped
  with the following iterations (see strategy_worker.py for why this is faithful);
* checkpoints: written by a background thread in the upstream formats.

usage: python train_fast.py CONFIG.json [--run-name NAME]   (or ./run.sh CONFIG.json ...)
Relative ``run_dir`` values resolve against this directory (``trainer/runs/...``,
ignored by git); the run dir must be new or empty.
"""
from __future__ import annotations

import os
import time


def _process_start_epoch() -> float:
    """Wall-clock time at which this process was created (from /proc)."""
    try:
        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/stat") as f:
            btime = next(int(l.split()[1]) for l in f if l.startswith("btime"))
        return btime + start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return time.time()


T_PROCESS = _process_start_epoch()
T_SCRIPT = time.time()

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import paths  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULTS = dict(
    n_iterations=250, n_traversals=30000, buffer_capacity=5_000_000, train_steps=10_000,
    strategy_train_steps=3000, batch_size=8192, lr=1e-3, checkpoint_every=10,
    seed=20261107, strategy_buffer_capacity=5_000_000, opponent_exploration_start=0.10,
    opponent_exploration_end=0.02, opponent_exploration_decay_iters=250,
    randomize_stacks=True, run_dir="runs/full",
    devices=dict(trav="cuda:1", fit="cuda:0", strat="cuda:2"),
    trav_precision="highest", trav_threads=24, trav_groups=2,
    fit_backend="cuda", fit_chunk=250, save_final_buffers=False,
    stop_after=None,  # timing runs: stop early, schedules still use n_iterations
)
N_PLAYERS, N_ACTIONS, FEATURE_DIM, TRUNK_DIM, HEAD_DIM = 6, 5, 51, 256, 128


def log(msg: str) -> None:
    print(msg, flush=True)


def lr_for(iteration: int, lr: float) -> float:
    """Upstream warm-up: 1e-4 + (lr - 1e-4) * t / 20 for t <= 20, else lr."""
    return 1e-4 + (lr - 1e-4) * (iteration / 20) if iteration <= 20 else lr


class Trainer:
    def __init__(self, cfg: dict, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        self.metrics_path = run_dir.parent / (run_dir.name + "_metrics.jsonl")
        self.timeline: dict[str, float] = {"process_start": T_PROCESS, "script_start": T_SCRIPT}

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        cfg = self.cfg
        t0 = time.time()
        seed = cfg["seed"]
        # original launch.py seeding (torch / numpy / random global streams)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        import gpu_fit as gf
        from ckpt import CheckpointWriter, params_from_state_dict
        from fit_backend import describe, make_fitter
        from gpu_traverse import BatchedTraverser
        from strategy_worker import StrategyWorker
        from training.network import DuelingAdvantageNet
        from training.train import _exploration_epsilon
        import common

        self.gf, self.params_from_state_dict = gf, params_from_state_dict
        self.eps_fn = _exploration_epsilon
        dev = cfg["devices"]
        self.fit_dev = torch.device(dev["fit"])

        # Iteration-1 traversal net: upstream builds DuelingAdvantageNet right after
        # torch.manual_seed(seed) with no torch RNG use in between, so this is the
        # exact same initial network as the original run.
        net0 = DuelingAdvantageNet(FEATURE_DIM, TRUNK_DIM, HEAD_DIM)
        self.params = common.params_from_torch(net0)
        self.adv_sd = {k: v.detach().clone() for k, v in net0.state_dict().items()}
        self.strat_sd = None  # upstream: untrained AverageStrategyNet until first fit

        self.writer = CheckpointWriter(self.run_dir)
        t1 = time.time()
        self.trav = BatchedTraverser(dev["trav"], cfg["trav_precision"],
                                     n_threads=cfg["trav_threads"], n_groups=cfg["trav_groups"])
        self.adv_set = gf.GPUReservoirSet(N_PLAYERS, cfg["buffer_capacity"], self.fit_dev,
                                          seed=seed + 1)
        self.strat_set = gf.GPUReservoirSet(1, cfg["strategy_buffer_capacity"], self.fit_dev,
                                            seed=seed + 2)
        self.adv_fitter = make_fitter("adv", self.adv_set, cfg["batch_size"], seed + 3,
                                      cfg["fit_backend"], cfg["fit_chunk"])
        self.strat_worker = StrategyWorker(dev["strat"], cfg["strategy_buffer_capacity"],
                                           cfg["batch_size"], seed + 4, cfg["fit_backend"],
                                           cfg["fit_chunk"])
        t2 = time.time()
        # Warm-ups outside the loop: numba JIT of the traversal, torch.compile +
        # CUDA-graph capture of both fitters.
        w = time.time()
        self.trav.run(self.params, 0, 64, 0.1, seed=12345)
        t_trav_warm = time.time() - w
        w = time.time()
        b = cfg["batch_size"]
        b = min(b, cfg["buffer_capacity"])
        self.adv_set.sizes = [b] * N_PLAYERS  # all-zero rows (weight 0): harmless
        self.adv_fitter.fit(1e-3, cfg["fit_chunk"])
        torch.cuda.synchronize(self.fit_dev)
        self.adv_set.sizes = [0] * N_PLAYERS
        t_adv_warm = time.time() - w
        t_strat_warm = self.strat_worker.warmup()
        self.timeline["setup_done"] = time.time()
        log(f"[setup] fit backend: {describe(cfg['fit_backend'])}")
        self.fit_manifest = paths.fit_manifest()
        log("[setup] fit code " + json.dumps(self.fit_manifest))
        log(f"[setup] imports+net0 {t1 - t0:.1f}s, allocate {t2 - t1:.1f}s, warm-up: traversal "
            f"{t_trav_warm:.1f}s, adv fitter {t_adv_warm:.1f}s, strat fitter {t_strat_warm:.1f}s; "
            f"process start -> loop start {self.timeline['setup_done'] - T_PROCESS:.1f}s")

    # ------------------------------------------------------------------- loop
    def _rng_meta(self) -> dict:
        meta = {"numpy": np.random.get_state(), "torch": torch.get_rng_state()}
        for name, gen in (("adv_reservoir", self.adv_set.gen), ("strat_reservoir", self.strat_set.gen),
                          ("adv_fitter", self.adv_fitter.gen),
                          ("strat_fitter", self.strat_worker.fitter.gen)):
            try:
                meta[name] = gen.get_state()
            except Exception as e:  # diagnostics must not kill the run
                meta[name] = repr(e)
        return meta

    def _checkpoint(self, it: int, lr: float, buf_sizes: list[int]) -> float:
        cfg = self.cfg
        t0 = time.time()
        adv_sd = self.adv_sd
        self.writer.write_advantage(it, adv_sd)
        meta = dict(
            iteration=it, buffer_sizes=buf_sizes, buffer_total=int(sum(buf_sizes)),
            strategy_buffer_size=len(self.strat_set),
            buffer_total_seen=list(self.adv_set.total_seen),
            strategy_total_seen=self.strat_set.total_seen[0],
            config=dict(cfg, architecture="dueling", deployment="advantage_npz",
                        feature_dim=FEATURE_DIM, trunk_dim=TRUNK_DIM, head_dim=HEAD_DIM,
                        n_actions=N_ACTIONS),
            seeds=dict(config_seed=cfg["seed"], traversal_seed=f"{cfg['seed']}*1000+iteration",
                       adv_reservoir=cfg["seed"] + 1, strat_reservoir=cfg["seed"] + 2,
                       adv_fitter=cfg["seed"] + 3, strat_fitter=cfg["seed"] + 4),
            rng_states=self._rng_meta(),
        )
        n_strat = len(self.strat_set)
        if n_strat >= 2 * cfg["batch_size"]:
            writer = self.writer

            def on_done(iteration, strat_sd, info, adv_sd=adv_sd, meta=meta):
                meta = dict(meta, strategy_fit=info)
                writer.write_strategy_and_pt(iteration, strat_sd, adv_sd, meta)

            self.strat_worker.submit(it, self.strat_set, lr, cfg["strategy_train_steps"], on_done)
        else:
            self.writer.write_strategy_and_pt(it, self.strat_sd, adv_sd, meta)
        return time.time() - t0

    def run(self) -> dict:
        cfg = self.cfg
        n_it, n_trav, batch = cfg["n_iterations"], cfg["n_traversals"], cfg["batch_size"]
        t_loop = time.time()
        self.timeline["loop_start"] = t_loop
        avg_loss = 0.0
        current_lr = cfg["lr"]
        mf = open(self.metrics_path, "a")
        last_it = n_it if cfg["stop_after"] is None else min(n_it, int(cfg["stop_after"]))
        for it in range(1, last_it + 1):
            t_it = time.time()
            traverser = (it - 1) % N_PLAYERS
            eps = self.eps_fn(it, n_it, cfg["opponent_exploration_start"],
                              cfg["opponent_exploration_end"], cfg["opponent_exploration_decay_iters"])
            # 1. traversal with the current advantage net
            res = self.trav.run(self.params, traverser, n_trav, eps, seed=cfg["seed"] * 1000 + it,
                                randomize_stacks=cfg["randomize_stacks"])
            t_a = time.time()
            # 2. reservoir insertion (upstream _collect_worker_batches semantics)
            feats, advs, sfeats, strats, sw = res.as_worker_result()
            n_samples = int(feats.shape[0])
            if n_samples:
                self.adv_set.add_batch(traverser, feats, advs, it)
            keep = sw > np.float32(1e-8)
            n_strat = int(keep.sum())
            if n_strat:
                self.strat_set.add_batch(0, sfeats[keep], strats[keep], it, sample_weights=sw[keep])
            torch.cuda.synchronize(self.fit_dev)
            t_b = time.time()
            # 3. advantage net: fresh init + 10k Adam steps when total >= 2*batch
            buf_sizes = list(self.adv_set.sizes)
            total = int(sum(buf_sizes))
            fitted = False
            if total >= batch * 2:
                current_lr = lr_for(it, cfg["lr"])
                fr = self.adv_fitter.fit(current_lr, cfg["train_steps"])
                avg_loss = fr.avg_loss
                self.adv_sd = fr.state_dict
                self.params = self.params_from_state_dict(fr.state_dict)
                fitted = True
            t_c = time.time()
            # 4. checkpoint: async exports + strategy fit on the spare GPU
            t_ck = 0.0
            if it % cfg["checkpoint_every"] == 0 or it == n_it or it == last_it:
                t_ck = self._checkpoint(it, current_lr, buf_sizes)
            t_end = time.time()

            it_time = t_end - t_it
            elapsed = t_end - t_loop
            eta = elapsed / it * (last_it - it) / 60
            last_strat = max(self.strat_worker.results) if self.strat_worker.results else None
            strat_loss = self.strat_worker.results[last_strat]["loss"] if last_strat else 0.0
            tm = res.timing
            rec = dict(iteration=it, traverser=traverser, adv_samples=n_samples,
                       strat_samples=n_strat, adv_loss=avg_loss, fitted=fitted,
                       strat_loss=strat_loss, strat_loss_iter=last_strat, lr=current_lr, eps=eps,
                       buffer=total, strategy_buffer=len(self.strat_set), buffer_sizes=buf_sizes,
                       t_traversal=t_a - t_it, t_add=t_b - t_a, t_fit=t_c - t_b, t_ckpt=t_ck,
                       t_iter=it_time, elapsed=elapsed, trav_nodes=res.n_nodes,
                       trav_rounds=res.n_rounds,
                       trav_split={k: round(v, 4) for k, v in tm.items()})
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            log(f"[Iter {it:4d}/{n_it}] traverser={traverser} adv_samples={n_samples:,} "
                f"strat_samples={n_strat:,} adv_loss={avg_loss:.4f} "
                f"strat_loss={strat_loss:.4f}(it{last_strat}) lr={current_lr:.1e} eps={eps:.3f} "
                f"buffer={total:,} strategy_buffer={len(self.strat_set):,} "
                f"[{','.join(f'{s // 1000}k' for s in buf_sizes)}] "
                f"speed={n_trav / it_time:.0f} trav/s time={it_time:.2f}s "
                f"(trav {t_a - t_it:.2f} add {t_b - t_a:.3f} fit {t_c - t_b:.2f} ckpt {t_ck:.3f}) "
                f"elapsed={elapsed:.0f}s ETA={eta:.1f}min")
        self.timeline["loop_compute_done"] = time.time()
        # drain: last strategy fit + all checkpoint writes belong to the run
        t_sw = self.strat_worker.wait()
        t_wr = self.writer.close()
        self.timeline["all_checkpoints_written"] = time.time()
        mf.close()
        errors = self.writer.errors + self.strat_worker.errors
        for e in errors:
            log("[error] " + e)
        summary = dict(
            iterations=last_it, fit_code=self.fit_manifest,
            loop_seconds_compute=self.timeline["loop_compute_done"] - t_loop,
            loop_seconds_incl_final_ckpt=self.timeline["all_checkpoints_written"] - t_loop,
            process_seconds_to_final_ckpt=self.timeline["all_checkpoints_written"] - T_PROCESS,
            setup_seconds=self.timeline["setup_done"] - T_PROCESS,
            final_drain_strategy_wait=t_sw, final_drain_writer_wait=t_wr,
            writer_busy_seconds=self.writer.busy_seconds,
            strategy_busy_seconds=self.strat_worker.busy_seconds,
            strategy_fits={str(k): v for k, v in sorted(self.strat_worker.results.items())},
            files_written=len(self.writer.written), errors=len(errors),
            timeline=self.timeline,
        )
        return summary

    def save_final_buffers(self) -> float:
        """Upstream-style latest.pt with full replay buffers (outside the timed run)."""
        t0 = time.time()
        bufs = []
        for b in range(N_PLAYERS):
            d = self.adv_set.state_dict(b)
            d.pop("sample_weights", None)  # upstream advantage buffers store none
            bufs.append(d)
        data = dict(iteration=self.cfg["n_iterations"], net=self.adv_sd,
                    strategy_net=self.writer_last_strat(), buffers=bufs,
                    strategy_buffer=self.strat_set.state_dict(0), config=self.cfg,
                    rng_states=self._rng_meta())
        path = self.run_dir / "latest.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(data, tmp)
        os.replace(tmp, path)
        return time.time() - t0

    def writer_last_strat(self):
        if not self.strat_worker.results:
            return None
        last = max(self.strat_worker.results)
        try:
            return torch.load(self.run_dir / f"iter_{last:05d}.pt", weights_only=False)["strategy_net"]
        except Exception:
            return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--run-name", default=None, help="overrides run_dir to runs/<name>")
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    user = json.loads(Path(args.config).read_text())
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    cfg.update(user)
    if args.run_name:
        cfg["run_dir"] = f"runs/{args.run_name}"
    run_dir = Path(cfg["run_dir"])
    if not run_dir.is_absolute():
        run_dir = paths.HERE / run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty run dir {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log("CONFIG " + json.dumps(cfg, sort_keys=True))
    log(f"[env] python {sys.version.split()[0]} torch {torch.__version__} numpy {np.__version__} "
        f"gpus {torch.cuda.device_count()} host {os.uname().nodename}")
    tr = Trainer(cfg, run_dir)
    tr.setup()
    summary = tr.run()
    log(f"\n[Deep CFR] Training complete. {summary['iterations']} iterations: loop "
        f"{summary['loop_seconds_compute']:.1f}s compute, "
        f"{summary['loop_seconds_incl_final_ckpt']:.1f}s incl. final strategy fit + writes; "
        f"whole process (start -> last checkpoint file) {summary['process_seconds_to_final_ckpt']:.1f}s "
        f"(setup {summary['setup_seconds']:.1f}s)")
    if cfg["save_final_buffers"]:
        secs = tr.save_final_buffers()
        summary["final_buffer_save_seconds"] = secs
        log(f"[final] replay buffers saved to {run_dir / 'latest.pt'} in {secs:.1f}s "
            f"(outside the timed run)")
    summary["process_seconds_total"] = time.time() - T_PROCESS
    (run_dir.parent / (run_dir.name + "_summary.json")).write_text(
        json.dumps(summary, indent=2, default=str) + "\n")
    log("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "timeline"}, default=str))
    if summary["errors"]:
        raise SystemExit(f"{summary['errors']} background errors; see log")


if __name__ == "__main__":
    main()
