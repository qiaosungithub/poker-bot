"""One isolated process per table; private state never leaves snapshot()."""
import contextlib
import json
import sys
import traceback

from .catalog import validate_config


def main():
    output = sys.stdout
    game = None
    version = 0
    with contextlib.redirect_stdout(sys.stderr):
        for line in sys.stdin:
            try:
                request = json.loads(line)
                command = request["command"]
                if command == "create":
                    config = validate_config(request["config"])
                    if config["mode"] == "classic":
                        from .classic import ClassicGame as Game
                    else:
                        from .bay import BayGame as Game
                    game = Game(config)
                elif game is None:
                    raise ValueError("请先创建牌桌。")
                elif command != "get" and request.get("version") != version:
                    raise ValueError("牌局已更新，请等待画面同步后再操作。")
                elif command == "action":
                    game.act(request["action"])
                elif command == "step":
                    game.step()
                elif command == "next":
                    game.next_hand()
                elif command == "fast":
                    for _ in range(500):
                        state = game.snapshot()
                        if state["done"] or state["actor"] == 0:
                            break
                        game.step()
                    else:
                        raise RuntimeError("Too many actions in one hand")
                elif command != "get":
                    raise ValueError("未知的牌桌操作。")
                if command != "get":
                    version += 1
                snapshot = game.snapshot()
                snapshot["version"] = version
                result = {"ok": True, "state": snapshot}
            except ValueError as error:
                result = {"ok": False, "error": str(error)}
            except Exception:
                traceback.print_exc()
                result = {"ok": False, "error": "牌局处理失败，请新建牌桌；服务日志中已保留原因。"}
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
