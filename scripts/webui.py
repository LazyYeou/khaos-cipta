"""Web UI lokal untuk upload gambar batik, konversi ke 64×64, training, dan sampling.

Hanya memakai library standar Python. UI menjalankan script yang sudah ada (prepare_patches.py,
train.py, sample.py) sebagai subprocess, satu job dalam satu waktu, dan menampilkan log-nya.

Contoh:
    python scripts/webui.py            # buka http://127.0.0.1:7860
    python scripts/webui.py --port 8000
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = Path(__file__).with_name("webui.html")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SERVED_DIRS = ("data/raw", "data/patches", "runs", "assets")
PREVIEW_LIMIT = 48


def safe_filename(name: str) -> str:
    """Nama file aman untuk disimpan: tanpa path, hanya karakter umum, ekstensi gambar."""
    base = Path(name.replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._") or "gambar"
    suffix = Path(base).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS - {".gif"}:
        raise ValueError(f"ekstensi tidak didukung: {suffix or '(kosong)'}")
    return stem + suffix


def resolve_served_path(root: Path, rel: str) -> Path | None:
    """Path file gambar di bawah folder yang boleh disajikan, atau None jika tidak diizinkan."""
    path = (root / unquote(rel)).resolve()
    if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
        return None
    for allowed in SERVED_DIRS:
        if path.is_relative_to((root / allowed).resolve()):
            return path
    return None


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


class JobRunner:
    """Menjalankan satu subprocess dalam satu waktu dan menyimpan ekor log-nya."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.kind = ""
        self.command = ""
        self.log: deque[str] = deque(maxlen=400)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, kind: str, args: list[str]) -> None:
        with self.lock:
            if self.running():
                raise RuntimeError(f"job '{self.kind}' masih berjalan")
            cmd = [sys.executable, "-u", *args]
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            self.kind, self.command = kind, " ".join(args)
            self.log.clear()
            self.log.append(f"$ python {self.command}")
            self.proc = subprocess.Popen(
                cmd,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _pump(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log.append(line.rstrip())
        proc.wait()
        self.log.append(f"[selesai, exit code {proc.returncode}]")

    def stop(self) -> None:
        with self.lock:
            if self.running():
                assert self.proc is not None
                self.proc.terminate()

    def state(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "running": self.running(),
            "returncode": None if self.proc is None else self.proc.poll(),
            "log": list(self.log),
        }


def collect_state(root: Path, jobs: JobRunner) -> dict[str, Any]:
    raw = list_images(root / "data/raw")
    patches = list_images(root / "data/patches")
    runs = []
    for run_dir in sorted((root / "runs").glob("*/")):
        samples = sorted((run_dir / "samples").glob("step_*.png"))
        extra = sorted(run_dir.glob("sample_*.png"), key=lambda p: p.stat().st_mtime)
        runs.append(
            {
                "name": run_dir.name,
                "checkpoint": (run_dir / "ema_last.pt").is_file(),
                "latest_sample": rel(root, samples[-1]) if samples else None,
                "generated": [rel(root, p) for p in extra],
            }
        )
    return {
        "raw": [{"name": str(p.relative_to(root / "data/raw")), "url": rel(root, p)} for p in raw],
        "patch_count": len(patches),
        "patch_preview": [rel(root, p) for p in patches[:PREVIEW_LIMIT]],
        "configs": sorted(p.name for p in (root / "configs").glob("*.yaml")),
        "runs": runs,
        "job": jobs.state(),
    }


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def build_convert_args(body: dict[str, Any]) -> list[str]:
    mode = body.get("mode", "auto")
    if mode not in ("auto", "resize", "patch"):
        raise ValueError("mode tidak valid")
    return [
        "scripts/prepare_patches.py",
        "--input", "data/raw",
        "--output", "data/patches",
        "--mode", mode,
        "--size", str(int(body.get("size", 64))),
        "--stride", str(int(body.get("stride", 32))),
        "--min-std", str(float(body.get("min_std", 0.05))),
        "--short-side", str(int(body.get("short_side", 256))),
    ]  # fmt: skip


def build_train_args(root: Path, body: dict[str, Any]) -> list[str]:
    config = Path(str(body.get("config", ""))).name
    if not (root / "configs" / config).is_file():
        raise ValueError("config tidak ditemukan")
    args = ["scripts/train.py", "--config", f"configs/{config}"]
    if body.get("smoke"):
        args.append("--smoke-test")
    if body.get("resume"):
        args.append("--resume")
    overrides = []
    if body.get("total_steps"):
        overrides.append(f"train.total_steps={int(body['total_steps'])}")
    if body.get("sample_every"):
        n = int(body["sample_every"])
        overrides += [f"train.sample_every={n}", f"train.save_every={n}"]
    if overrides:
        args += ["--override", *overrides]
    return args


def build_sample_args(root: Path, body: dict[str, Any]) -> list[str]:
    run = Path(str(body.get("run", ""))).name
    ckpt = root / "runs" / run / "ema_last.pt"
    if not run or not ckpt.is_file():
        raise ValueError("checkpoint tidak ditemukan")
    sampler = body.get("sampler", "ddim")
    if sampler not in ("ddpm", "ddim"):
        raise ValueError("sampler tidak valid")
    return [
        "scripts/sample.py",
        "--ckpt", f"runs/{run}/ema_last.pt",
        "--sampler", sampler,
        "--steps", str(int(body.get("steps", 50))),
        "--n", str(int(body.get("n", 16))),
        "--seed", str(int(body.get("seed", 0))),
    ]  # fmt: skip


class Handler(BaseHTTPRequestHandler):
    server: "WebUIServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # jangan penuhi terminal dengan log setiap request polling

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self.send_json({"error": message}, status)

    def read_body(self, limit: int) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if length > limit:
            raise ValueError(f"ukuran request melebihi {limit // (1024 * 1024)} MB")
        return self.rfile.read(length)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        root = self.server.root
        if url.path == "/":
            body = HTML_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/state":
            self.send_json(collect_state(root, self.server.jobs))
        elif url.path.startswith("/files/"):
            path = resolve_served_path(root, url.path.removeprefix("/files/"))
            if path is None:
                self.send_error_json("file tidak ditemukan", HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", CONTENT_TYPES[path.suffix.lower()])
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error_json("tidak ditemukan", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        root, jobs = self.server.root, self.server.jobs
        try:
            if url.path == "/api/upload":
                name = safe_filename(parse_qs(url.query).get("name", [""])[0])
                data = self.read_body(MAX_UPLOAD_BYTES)
                raw_dir = root / "data/raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / name).write_bytes(data)
                self.send_json({"saved": name})
                return

            body = json.loads(self.read_body(1024 * 1024) or b"{}")
            if url.path == "/api/delete-raw":
                path = resolve_served_path(root, f"data/raw/{body.get('name', '')}")
                if path is None or not path.is_relative_to((root / "data/raw").resolve()):
                    raise ValueError("file tidak ditemukan")
                path.unlink()
            elif url.path == "/api/convert":
                if jobs.running():
                    raise RuntimeError(f"job '{jobs.kind}' masih berjalan")
                if body.get("clear"):
                    shutil.rmtree(root / "data/patches", ignore_errors=True)
                jobs.start("convert", build_convert_args(body))
            elif url.path == "/api/train":
                jobs.start("train", build_train_args(root, body))
            elif url.path == "/api/sample":
                jobs.start("sample", build_sample_args(root, body))
            elif url.path == "/api/stop":
                jobs.stop()
            else:
                self.send_error_json("tidak ditemukan", HTTPStatus.NOT_FOUND)
                return
            self.send_json({"ok": True})
        except (ValueError, RuntimeError, json.JSONDecodeError) as e:
            self.send_error_json(str(e))


class WebUIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], root: Path) -> None:
        super().__init__(address, Handler)
        self.root = root.resolve()
        self.jobs = JobRunner(self.root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1", help="default hanya bisa diakses lokal")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="jangan buka browser otomatis")
    args = parser.parse_args()

    server = WebUIServer((args.host, args.port), REPO_ROOT)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"Batik Diffusion UI berjalan di {url} (Ctrl+C untuk berhenti)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.jobs.stop()
        server.server_close()


if __name__ == "__main__":
    main()
