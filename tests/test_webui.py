import importlib.util
import io
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "webui", Path(__file__).resolve().parents[1] / "scripts" / "webui.py"
)
webui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(webui)


def png_bytes(w: int = 10, h: int = 8) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 60, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_safe_filename_strips_paths_and_odd_characters() -> None:
    assert webui.safe_filename("../../etc/kawung motif.PNG") == "kawung_motif.png"
    assert webui.safe_filename("C:\\Users\\x\\parang.jpg") == "parang.jpg"
    with pytest.raises(ValueError):
        webui.safe_filename("script.py")


def test_resolve_served_path_blocks_traversal(tmp_path: Path) -> None:
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/a.png").write_bytes(png_bytes())
    (tmp_path / "secret.png").write_bytes(png_bytes())

    assert webui.resolve_served_path(tmp_path, "data/raw/a.png") is not None
    assert webui.resolve_served_path(tmp_path, "data/raw/../../secret.png") is None
    assert webui.resolve_served_path(tmp_path, "secret.png") is None
    assert webui.resolve_served_path(tmp_path, "data/raw/missing.png") is None


def test_build_args_validate_input(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/batik64.yaml").write_text("seed: 0\n")

    args = webui.build_train_args(
        tmp_path, {"config": "batik64.yaml", "smoke": True, "total_steps": 30, "sample_every": 10}
    )
    assert args == [
        "scripts/train.py", "--config", "configs/batik64.yaml", "--smoke-test",
        "--override", "train.total_steps=30", "train.sample_every=10", "train.save_every=10",
    ]  # fmt: skip
    with pytest.raises(ValueError):
        webui.build_train_args(tmp_path, {"config": "../pyproject.toml"})
    with pytest.raises(ValueError):
        webui.build_sample_args(tmp_path, {"run": "tidak_ada"})
    with pytest.raises(ValueError):
        webui.build_convert_args({"mode": "rm -rf"})
    assert "--mode" in webui.build_convert_args({"mode": "resize"})


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/batik64.yaml").write_text("seed: 0\n")
    srv = webui.WebUIServer(("127.0.0.1", 0), tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def request(url: str, data: bytes | None = None) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data)) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_upload_list_serve_and_delete(server: str) -> None:
    status, _ = request(f"{server}/api/upload?name=kawung.png", png_bytes())
    assert status == 200

    state = json.loads(request(f"{server}/api/state")[1])
    assert [r["name"] for r in state["raw"]] == ["kawung.png"]
    assert state["configs"] == ["batik64.yaml"]
    assert state["job"]["running"] is False

    status, body = request(f"{server}/files/data/raw/kawung.png")
    assert status == 200 and body == png_bytes()
    assert request(f"{server}/files/configs/batik64.yaml")[0] == 404

    status, _ = request(f"{server}/api/delete-raw", json.dumps({"name": "kawung.png"}).encode())
    assert status == 200
    assert json.loads(request(f"{server}/api/state")[1])["raw"] == []


def test_upload_rejects_non_image(server: str) -> None:
    status, body = request(f"{server}/api/upload?name=run.py", b"print(1)")
    assert status == 400
    assert "ekstensi" in json.loads(body)["error"]


def test_index_page_is_served(server: str) -> None:
    status, body = request(f"{server}/")
    assert status == 200
    assert b"<title>Batik Diffusion</title>" in body
