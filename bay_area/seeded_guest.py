"""SDK guest with reproducible, per-request PRNG seeds; bot files stay unchanged.

The wrapper does not replace the bots' time guards or strategy parameters.
Its overhead is inside the real SDK host deadline. Public table/request context
and memory determine seeds, so worker ordering cannot change the random stream.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

from poker_agent_sdk.guest import load, serve


def random_generators(module, agent, folder):
    found, visited = [], set()

    def visit(obj):
        if id(obj) in visited:
            return
        visited.add(id(obj))
        for name, value in sorted(vars(obj).items()):
            if isinstance(value, random.Random):
                if id(value) not in visited:
                    visited.add(id(value))
                    found.append(value)
            elif isinstance(value, ModuleType):
                file = getattr(value, "__file__", None)
                if file and Path(file).resolve().is_relative_to(folder):
                    visit(value)
    visit(module)
    visit(agent)
    return found


class SeededAgent:
    def __init__(self, module, folder):
        self.agent = module.Agent()
        self.generators = random_generators(module, self.agent, folder)

    def act(self, state, memory):
        context = (f"{state.table_id if hasattr(state, 'table_id') else state.config.table_id}:"
                   f"{state.seat}:{state.hand_number}:{state.phase}:{len(state.history)}").encode()
        digest = hashlib.blake2b(context + memory, digest_size=16).digest()
        seed = int.from_bytes(digest, "little")
        random.seed(seed)
        for index, generator in enumerate(self.generators):
            generator.seed(seed + index + 1)
        return self.agent.act(state, memory)


def main():
    folder = Path(sys.argv[1]).resolve()
    output = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    module = load(str(folder))
    if isinstance(module, BaseException):
        wrapped = module
    else:
        wrapped = SimpleNamespace(Agent=lambda: SeededAgent(module, folder))
    serve(wrapped, sys.stdin.buffer, output)


if __name__ == "__main__":
    main()
