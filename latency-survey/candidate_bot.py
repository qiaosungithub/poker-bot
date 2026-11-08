"""Existing Fullhouse 6-max policy with FP32 ONNX inference.

Input/output schema: fullhouse-bot/engine_vendored/bots/template/bot.py.
This is NOT the unreleased No Limit Agent interface and is not an 8-player bot.
Measured sub-millisecond p99, but rare >1ms calls were observed; see JSON results.
Upstream policy: advitrocks9/fullhouse-bot, commit e504793d480b1b975f25258d25939b45c6dbd5a4.
Upstream MIT license is retained in fullhouse-bot/LICENSE.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "fullhouse-bot"))
from bot import bot as _bot
from fullhouse_onnx import ONNXLookup

_bot._DEEP_CFR = ONNXLookup(_ROOT / "fullhouse_fp32.onnx")
decide = _bot.decide

if __name__ == "__main__":
    import json
    for _line in sys.stdin:
        print(json.dumps(decide(json.loads(_line))), flush=True)
