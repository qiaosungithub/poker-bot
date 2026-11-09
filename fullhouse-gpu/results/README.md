# Measured records of the GPU reproduction

Copied unchanged from `qiaos-4a100-2:~/work/poker-fast/` on 2026-11-08. File
contents still name the VM paths they were written under
(`/home/qiaos/work/poker-fast/trainer/...`); they are historical records, not
paths the code uses. Checkpoints, ONNX exports and replay buffers are not in Git;
their SHA256 values are listed below.

Run names: **run 1** = trainer run `full` (fit kernels release v1),
**run 2** = trainer run `full_v2` (fit kernels release v2, the code in `../fit/`).
Both use the same configuration and seeds (`../trainer/configs/full.json`,
`full_v2.json`; only `run_dir` differs).

| Path | Content |
|---|---|
| `runs/full_metrics.jsonl`, `runs/full_v2_metrics.jsonl` | One record per iteration: samples, losses, learning rate, epsilon, reservoir sizes, time split (`t_traversal`, `t_add`, `t_fit`, `t_ckpt`, `t_iter`) and the traversal's internal split |
| `runs/full_summary.json`, `runs/full_v2_summary.json` | Whole-run timings (loop, process start to last checkpoint file, setup, final buffer save), strategy fits, and `fit_code`: SHA256 of every fitting source file the run imported |
| `runs/checkpoints_full.sha256`, `runs/checkpoints_full_v2.sha256` | SHA256 of every advantage NPZ (`*_adv/iter_*.npz`, the deployed bot path), strategy NPZ and `.pt` checkpoint; size and time of the 9.2 GB `latest.pt` |
| `runs/{smoke,timing10,vcheck,vcheck_v2}_*` | Validation runs before the real runs (4-iteration smoke, 10-iteration full-scale timing, 3-iteration checks of both fit versions) |
| `logs/` | Console logs of all trainer runs (`full.log` = run 1, `full_v2.log` = run 2), the eval drivers and the eval-venv installation |
| `eval/full/`, `eval/full_v2/` | Six-player protocol outputs of runs 1 and 2: `status.json` (selected checkpoint, confirmation, criterion), `results.jsonl` (one record per arena), per-checkpoint `iter_*_selection.json` + raw per-block `.jsonl`, `iter_*_confirmation.json(l)`, `reference_*.json(l)` (released model on the same seeds), and `iter_*.onnx.validation.json` (ONNX vs NumPy bot on 1,024 states) |
| `traversal/` | Traversal validation and benchmarks behind `../reports/traversal_REPORT.md`: `exact_*.json` (bit-exactness / tolerance vs the recursive reference), `stats_*.json` (149 statistics vs unmodified upstream), `bench_*.json` (timings), suite logs |
| `fit/` | `validate_adv_cuda.json`, `validate_strat_cuda.json`, `validate.log`: 3-seed full fits of the v2 GPU fitter vs the fp32 JAX reference on a realistic buffer (`../reports/fit_REPORT.md`) |

The `fit_code` hashes in `runs/full_v2_summary.json` equal `../fit/MANIFEST.sha256`
(run 2 trained with exactly the committed fitting code). Run 1's v1 kernels
(`gpu_fit.py` `1c1eeea9…`, `fast_step.py` `e61e57bd…`, `rowk.cu` `8f7ec1c3…`,
`triton_fit3.py` `33f75eef…`) are recorded only by these hashes in
`runs/full_summary.json`; the source stayed on the VM
(`~/work/poker-fast/trainer/vendor_fit/`).

The reference arenas of runs 1 and 2 (`eval/*/reference_*.jsonl`) are
byte-identical to each other, and their means equal the original reproduction's
reference values (selection 59.18956, confirmation 58.5894).
