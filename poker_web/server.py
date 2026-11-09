"""Loopback practice server; run with python -m poker_web.server."""
import argparse
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import secrets
import selectors
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

from .catalog import ROOT, catalog, validate_config

STATIC = Path(__file__).parent / "static"
RUNTIME = ROOT / ".runtime/web"
PYTHONS = {
    "classic": ROOT / ".runtime/envs/poker-hu-validation/bin/python",
    "bay_area": ROOT / ".runtime/envs/poker-bay-area312/bin/python",
}


class TableProcess:
    def __init__(self, config):
        self.id = secrets.token_urlsafe(20)
        self.lock = threading.Lock()
        self.touched = time.monotonic()
        (RUNTIME / "logs").mkdir(parents=True, exist_ok=True)
        log = (RUNTIME / "logs" / (self.id + ".log")).open("a")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", TMPDIR=str(RUNTIME / "tmp"))
        (RUNTIME / "tmp").mkdir(exist_ok=True)
        try:
            self.process = subprocess.Popen([str(PYTHONS[config["mode"]]), "-u", "-m", "poker_web.worker"],
                                            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=log, text=True, bufsize=1)
        finally:
            log.close()
        try:
            result = self.call({"command": "create", "config": config})
            if not result["ok"]:
                raise ValueError(result["error"])
            self.initial = result
        except Exception:
            self.close()
            raise

    def call(self, data):
        with self.lock:
            self.touched = time.monotonic()
            if self.process.poll() is not None:
                raise ValueError("牌桌进程已停止，请新建牌桌。")
            self.process.stdin.write(json.dumps(data) + "\n")
            self.process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                if not selector.select(30):
                    self.close()
                    raise ValueError("策略响应超时，请新建牌桌。")
            line = self.process.stdout.readline()
            if not line:
                raise ValueError("策略进程意外停止，请新建牌桌。")
            result = json.loads(line)
            if result.get("state"):
                result["state"]["game_id"] = self.id
            return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address):
        super().__init__(address, Handler)
        self.games = {}
        self.guard = threading.RLock()

    def prune(self):
        with self.guard:
            for owner, game in list(self.games.items()):
                if time.monotonic() - game.touched > 10800:
                    game.close()
                    del self.games[owner]

    def server_close(self):
        with self.guard:
            for game in self.games.values():
                game.close()
            self.games.clear()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = "PokerPractice"
    def owner(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            owner = cookie["poker_session"].value
            if len(owner) != 43:
                raise ValueError()
            return owner, False
        except (KeyError, ValueError):
            return secrets.token_urlsafe(32), True

    def respond(self, status, data, content_type="application/json; charset=utf-8"):
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False).encode()
        elif isinstance(data, str):
            data = data.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        if self.new_cookie:
            self.send_header("Set-Cookie", f"poker_session={self.session}; HttpOnly; SameSite=Strict; Path=/; Max-Age=10800")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.session, self.new_cookie = self.owner()
        path = urlparse(self.path).path
        try:
            if path == "/api/catalog":
                self.respond(200, {"modes": catalog()})
            elif path == "/api/session":
                with self.server.guard:
                    game = self.server.games.get(self.session)
                self.respond(200, game.call({"command": "get"}) if game else {"ok": True, "state": None})
            elif path == "/health":
                self.respond(200, {"ok": True})
            else:
                names = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/favicon.svg": "favicon.svg"}
                if path not in names:
                    self.respond(404, {"error": "页面不存在。"})
                    return
                file = STATIC / names[path]
                kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                self.respond(200, file.read_bytes(), kind + ("; charset=utf-8" if kind.startswith("text/") else ""))
        except (ValueError, OSError) as error:
            self.respond(503, {"ok": False, "error": str(error)})

    def do_POST(self):
        self.session, self.new_cookie = self.owner()
        origin = self.headers.get("Origin")
        if (origin and urlparse(origin).netloc != self.headers.get("Host")) or "application/json" not in self.headers.get("Content-Type", ""):
            self.respond(403, {"ok": False, "error": "请求来源无效。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("请求大小无效。")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("请求格式无效。")
            path = urlparse(self.path).path
            if path == "/api/game":
                config = validate_config(data)
                self.server.prune()
                with self.server.guard:
                    if len(self.server.games) >= 8 and self.session not in self.server.games:
                        raise ValueError("牌桌已满，请稍后再试。")
                    game = TableProcess(config)
                    previous = self.server.games.get(self.session)
                    self.server.games[self.session] = game
                    if previous:
                        previous.close()
                self.respond(200, game.initial)
            elif path == "/api/command":
                if data.get("command") not in ("action", "step", "next", "fast"):
                    raise ValueError("未知的操作。")
                with self.server.guard:
                    game = self.server.games.get(self.session)
                if not game or data.get("game_id") != game.id:
                    raise ValueError("牌桌已过期或已在其他页面更换，请刷新页面。")
                result = game.call(data)
                self.respond(200 if result["ok"] else 400, result)
            elif path == "/api/leave":
                with self.server.guard:
                    game = self.server.games.get(self.session)
                    if game and data.get("game_id") == game.id:
                        self.server.games.pop(self.session)
                        game.close()
                self.respond(200, {"ok": True})
            else:
                self.respond(404, {"ok": False, "error": "接口不存在。"})
        except (ValueError, TypeError, KeyError) as error:
            self.respond(400, {"ok": False, "error": str(error)})
        except (OSError, BrokenPipeError):
            self.respond(503, {"ok": False, "error": "牌桌进程不可用，请新建牌桌。"})

    def log_message(self, fmt, *args):
        print(f"{self.log_date_time_string()} {fmt % args}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    server = Server((args.host, args.port))
    print(f"Poker practice ready: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
