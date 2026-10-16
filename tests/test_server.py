import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from pickplace.server import make_handler


@pytest.fixture
def server(tmp_path):
    (tmp_path / "web").mkdir()
    (tmp_path / "web/index.html").write_text("<h1>Test</h1>")
    (tmp_path / "secret.txt").write_text("must not be served")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_health_and_static_files(server):
    with urlopen(server + "/api/health") as response:
        assert json.load(response) == {"status": "ok"}
    with urlopen(server + "/") as response:
        assert b"Test" in response.read()
        assert response.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize(
    "path,code",
    [
        ("/../secret.txt", 404),
        ("/.venv/pyvenv.cfg", 404),
        ("/api/rollout?policy=../../secret", 400),
        ("/api/rollout?seed=-1", 400),
        ("/api/rollout?seed=abc", 400),
        ("/api/rollout?distribution=invalid", 400),
        ("/api/rollout?policy=bc256", 404),
    ],
)
def test_server_input_validation_and_file_allowlist(server, path, code):
    with pytest.raises(HTTPError) as exc:
        urlopen(server + path)
    assert exc.value.code == code
    assert "error" in json.load(exc.value)


def test_fresh_rollout_endpoint(server):
    with urlopen(server + "/api/rollout?policy=expert&seed=42000") as response:
        data = json.load(response)
    assert data["success"]
    assert data["id"] == "fresh-expert-id-42000"
    assert len(data["frames"]) > 1
