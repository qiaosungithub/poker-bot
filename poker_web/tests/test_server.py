import http.cookiejar
import json
import threading
import unittest
import urllib.error
import urllib.request

from poker_web.server import Server


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = Server(("127.0.0.1", 0))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def client(self):
        return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def request(self, client, path, body=None, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        req = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        try:
            with client.open(req) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_sessions_stale_actions_and_origin_boundary(self):
        a, b = self.client(), self.client()
        self.request(a, "/api/catalog")
        self.request(b, "/api/catalog")
        code, reply = self.request(a, "/api/game", dict(mode="classic", bots=["solver_like"], name="<test>"))
        self.assertEqual(code, 200)
        state = reply["state"]
        cmd = dict(command="step", game_id=state["game_id"], version=state["version"])
        self.assertEqual(self.request(b, "/api/command", cmd)[0], 400)
        self.assertIsNone(self.request(b, "/api/session")[1]["state"])
        self.assertEqual(self.request(a, "/api/command", cmd, origin="https://example.org")[0], 403)
        cmd["version"] -= 1
        self.assertEqual(self.request(a, "/api/command", cmd)[0], 400)
        self.assertEqual(self.request(a, "/api/session")[1]["state"], state)
        self.assertEqual(self.request(a, "/api/game", dict(mode="bay_area", bots=["fullhouse_released"], name="x"))[0], 400)
        self.assertEqual(self.request(a, "/api/session")[1]["state"], state)
        self.request(a, "/api/leave", dict(game_id=state["game_id"]))
        self.assertIsNone(self.request(a, "/api/session")[1]["state"])


if __name__ == "__main__":
    unittest.main()
